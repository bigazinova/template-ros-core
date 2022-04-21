import functools

import cv2
import numpy as np
from math import sqrt
from line_detector_interface import LineDetectorInterface
from .detections import Detections

MIN_PIXEL = 3
MAX_PIXEL = 200
PART_OF_INTEREST = 0.7
EPS = 8
CAMERA_MATRIX = np.array(
    [[278.79547761007365, 0.0, 314.29374336264345], [0.0, 280.52395701002115, 228.59132685202135], [0.0, 0.0, 1.0]])
DISTORTION_COEFFFICIENTS = np.array(
    [-0.23670917420627122, 0.03455456424406622, 0.0037778674941860426, 0.0020245279929775382, 0.0])


class LineDetector(LineDetectorInterface):
    """
    The Line Detector can be used to extract line segments from a particular color range in an image. It combines
    edge detection, color filtering, and line segment extraction.

    This class was created for the goal of extracting the white, yellow, and red lines in the Duckiebot's camera stream
    as part of the lane localization pipeline. It is setup in a way that allows efficient detection of line segments in
    different color ranges.

    In order to process an image, first the :py:meth:`setImage` method must be called. In makes an internal copy of the
    image, converts it to `HSV color space <https://en.wikipedia.org/wiki/HSL_and_HSV>`_, which is much better for
    color segmentation, and applies `Canny edge detection <https://en.wikipedia.org/wiki/Canny_edge_detector>`_.

    Then, to do the actual line segment extraction, a call to :py:meth:`detectLines` with a :py:class:`ColorRange`
    object must be made. Multiple such calls with different colour ranges can be made and these will reuse the
    precomputed HSV image and Canny edges.

    Args:

        canny_thresholds (:obj:`list` of :obj:`int`): a list with two entries that specify the thresholds for the hysteresis procedure, details `here <https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=canny#canny>`__, default is ``[80, 200]``

        canny_aperture_size (:obj:`int`): aperture size for a Sobel operator, details `here <https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=canny#canny>`__, default is 3

        dilation_kernel_size (:obj:`int`): kernel size for the dilation operation which fills in the gaps in the color filter result, default is 3

        hough_threshold (:obj:`int`): Accumulator threshold parameter. Only those lines are returned that get enough votes, details `here <https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghlinesp#houghlinesp>`__, default is 2

        hough_min_line_length (:obj:`int`): Minimum line length. Line segments shorter than that are rejected, details `here <https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghlinesp#houghlinesp>`__, default is 3

        hough_max_line_gap (:obj:`int`): Maximum allowed gap between points on the same line to link them, details `here <https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghlinesp#houghlinesp>`__, default is 1

    """

    class Element:
        def __init__(self, ncenter, ndx, ndy, rad):
            self.center = ncenter
            self.dx = int(ndx)
            self.dy = int(ndy)
            self.rad = int(rad)
            

    def __init__(
        self,
        canny_thresholds=[80, 200],
        canny_aperture_size=3,
        dilation_kernel_size=3,
        hough_threshold=2,
        hough_min_line_length=3,
        hough_max_line_gap=1,
    ):

        self.canny_thresholds = canny_thresholds
        self.canny_aperture_size = canny_aperture_size
        self.dilation_kernel_size = dilation_kernel_size
        self.hough_threshold = hough_threshold
        self.hough_min_line_length = hough_min_line_length
        self.hough_max_line_gap = hough_max_line_gap

        # initialize the variables that will hold the processed images
        self.bgr = np.empty(0)  #: Holds the ``BGR`` representation of an image
        self.hsv = np.empty(0)  #: Holds the ``HSV`` representation of an image
        self.canny_edges = np.empty(0)  #: Holds the Canny edges of an image
        self.last_dash_line = []

    @staticmethod
    def get_max_dist_between_elements(img, y):
        height, width, _ = img.shape
        max_dist = 170
        step = 100
        if height - step <= y <= height:
            return max_dist
        if height - step * 2 <= y:
            return (max_dist >> 1) + 10
        elif height - step * 2.5 <= y:
            return max_dist >> 2
        else:
            return max_dist >> 5

    @staticmethod
    def _max_radius(h, img_len):
        if h > img_len * PART_OF_INTEREST:
            return 0
        alpha = (1 - (MIN_PIXEL + EPS) / MAX_PIXEL) / (img_len * PART_OF_INTEREST)
        return MAX_PIXEL * (1 - alpha * h)

    @staticmethod
    def _dist_pixels(pix1, pix2):
        return int(sqrt((pix1[0] - pix2[0]) ** 2 + (pix1[1] - pix2[1]) ** 2))

    def _filter_contours(self, contours, img):

        filtered_contours = []
        for k, contour in enumerate(contours):
            approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
            center = sum(approx) / approx.shape[0]

            radius = max(contour, key=lambda x: abs(center[0][0] - x[0][0]) ** 2 + abs(center[0][1] - x[0][1]) ** 2)

            radius_length = ((radius[0][0] - center[0][0]) ** 2 + (radius[0][1] - center[0][1]) ** 2) ** (0.5)
            if radius_length > 60:
                continue

            if not MIN_PIXEL < radius_length < MAX_PIXEL:
                continue

            if len(approx) in (3, 4):
                # cv2.drawContours(img, contours, k, (0, 100, 100), thickness=2)
                dx = 0
                dy = 0
                for el in approx:
                    x1 = int(el[0][0])
                    y1 = int(el[0][1])
                    for el2 in approx:
                        x2 = int(el2[0][0])
                        y2 = int(el2[0][1])
                        dx = max(dx, abs(x2 - x1))
                        dy = max(dy, abs(y2 - y1))
                dy = max(dy, int(dx * 0.4))
                dx //= 2
                dy //= 2
                ans = []
                max_dist = 0

                for el in approx:
                    for el2 in approx:
                        if self._dist_pixels(el[0], el2[0]) > max_dist:
                            max_dist = self._dist_pixels(el[0], el2[0])
                            ans = [el[0], el2[0]]
                ncent = [(ans[0][0] + ans[1][0]) // 2, (ans[0][1] + ans[1][1]) // 2]

                if np.array_equal(img[int(center[0][1]), int(center[0][0])], [0, 0, 0]):

                    continue
                # cv2.drawContours(img, contours, k, (100, 0, 100), thickness=1)

                filtered_contours.append(self.Element(ncent, dx, dy, radius_length))
                
        return filtered_contours


    def sort_contours(self, contours):
        return sorted(contours, key=functools.cmp_to_key(lambda contour1, contour2: self._dist_pixels(contour1.center, contour2.center)), reverse=True)
        # int(sqrt((pix1[0] - pix2[0]) ** 2 + (pix1[1] - pix2[1]) ** 2))
    # def detect_dash_line_for_pub(self, img):
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     ret, threshold_image = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)  # 150, 255, 0)
    #     contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     contours = self._filter_contours(contours, threshold_image)
    #     contours = self.sort_contours(contours)
    #     for i, contour in enumerate(contours):
    #         cv2.circle(threshold_image, (int(contour.center[0]), int(contour.center[1])), int(contour.rad), (255, 255, 0),
    #                    thickness=5)
    #     # next_contour = False
    #     # ans_dash_line = []
    #     # prev_contour = contours[0]
    #     # cv2.circle(threshold_image, (int(prev_contour.center[0]), int(prev_contour.center[1])), int(prev_contour.rad), (255, 255, 0),
    #     #            thickness=3)
    #     return threshold_image

    def _detect_dash_line(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, threshold_image = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)  # 150, 255, 0)
        contours, h = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        threshold_image = cv2.cvtColor(threshold_image, cv2.COLOR_GRAY2RGB)

        contours = self._filter_contours(contours, threshold_image)
        contours = self.sort_contours(contours)
        next_contour = False
        ans_dash_line = []

        prev_contour = contours[0]
        # cv2.circle(threshold_image, (int(prev_contour.center[0]), int(prev_contour.center[1])), int(prev_contour.rad), (255, 255, 0),
        #                thickness=1)

        cv2.putText(threshold_image, str(0), (int(prev_contour.center[0]), int(prev_contour.center[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 8, 230), thickness=1)
        for i, contour in enumerate(contours[1:]):
            # cv2.circle(threshold_image, (int(contour.center[0]), int(contour.center[1])), int(contour.rad), (255, 255, 0),
            #                thickness=1)
            max_dist = self.get_max_dist_between_elements(threshold_image, prev_contour.center[1])
            print(i)
            print(max_dist)

            cv2.putText(threshold_image, str(i+1), (int(contour.center[0]), int(contour.center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 8, 230), thickness=1)
            dist = self._dist_pixels(contour.center, prev_contour.center)
            print(dist)
            print(self.is_inside(contour, prev_contour))
            if dist <= max_dist and not self.is_inside(contour, prev_contour):
                ans_dash_line.append(prev_contour)
                cv2.circle(threshold_image, (int(prev_contour.center[0]), int(prev_contour.center[1])), int(prev_contour.rad), (255, 255, 0),
                           thickness=1)
                next_contour = True
            else:
                if next_contour:
                    ans_dash_line.append(prev_contour)
                    cv2.circle(threshold_image, (int(prev_contour.center[0]), int(prev_contour.center[1])), int(prev_contour.rad), (255, 255, 0),
                                              thickness=1)
                next_contour = False

            prev_contour = contour
            print('-'*80)

        if next_contour:
            ans_dash_line.append(prev_contour)
            cv2.circle(threshold_image, (int(prev_contour.center[0]), int(prev_contour.center[1])), int(prev_contour.rad), (255, 255, 0),
                       thickness=1)
        return threshold_image, ans_dash_line

    @staticmethod
    def is_inside(first_line, second_line):
        if first_line.center == second_line.center and first_line.rad == second_line.rad:  # переопределить метод сравнения
            return False
        big_cont = first_line if first_line.rad > second_line.rad else second_line
        small_cont = first_line if first_line.rad < second_line.rad else second_line
        dist_between_cent = LineDetector._dist_pixels(first_line.center, second_line.center)
        if small_cont.rad + dist_between_cent < big_cont.rad:
            return True
        return False

    @staticmethod
    def is_cross(first_line, second_line):
        dist_between_cent = LineDetector._dist_pixels(first_line.center, second_line.center)
        e = 5 # зависимость от координат
        if first_line.rad + second_line.rad < dist_between_cent + e:
            return False
        return True

    @staticmethod
    def _make_undistorted_image(img):
        return cv2.undistort(img, CAMERA_MATRIX, DISTORTION_COEFFFICIENTS)

    @staticmethod
    def _sum(pix1, pix2):
        return [pix1[0] + pix2[0], pix1[1] + pix2[1]]


    def setImage(self, image):
        """
        Sets the :py:attr:`bgr` attribute to the provided image. Also stores
        an `HSV <https://en.wikipedia.org/wiki/HSL_and_HSV>`_ representation of the image and the
        extracted `Canny edges <https://en.wikipedia.org/wiki/Canny_edge_detector>`_. This is separated from
        :py:meth:`detectLines` so that the HSV representation and the edge extraction can be reused for multiple
        colors.

        Args:
            image (:obj:`numpy array`): input image

        """

        self.bgr = np.copy(image)
        self.hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.canny_edges = self.findEdges()

    def getImage(self):
        """
        Provides the image currently stored in the :py:attr:`bgr` attribute.

        Returns:
            :obj:`numpy array`: the stored image
        """
        return self.bgr

    def findEdges(self):
        """
        Applies `Canny edge detection <https://en.wikipedia.org/wiki/Canny_edge_detector>`_ to a ``BGR`` image.


        Returns:
            :obj:`numpy array`: a binary image with the edges
        """
        edges = cv2.Canny(
            self.bgr,
            self.canny_thresholds[0],
            self.canny_thresholds[1],
            apertureSize=self.canny_aperture_size,
        )
        return edges

    def houghLine(self, edges):
        """
        Finds line segments in a binary image using the probabilistic Hough transform. Based on the OpenCV function
        `HoughLinesP <https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghlinesp
        #houghlinesp>`_.

        Args:
            edges (:obj:`numpy array`): binary image with edges

        Returns:
             :obj:`numpy array`: An ``Nx4`` array where each row represents a line ``[x1, y1, x2, y2]``. If no lines
             were detected, returns an empty list.

        """
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap,
        )
        if lines is not None:
            lines = lines.reshape((-1, 4))  # it has an extra dimension
        else:
            lines = []

        return lines

    def colorFilter(self, color_range):
        """
        Obtains the regions of the image that fall in the provided color range and the subset of the detected Canny
        edges which are in these regions. Applies a `dilation <https://homepages.inf.ed.ac.uk/rbf/HIPR2/dilate.htm>`_
        operation to smooth and grow the regions map.

        Args:
            color_range (:py:class:`ColorRange`): A :py:class:`ColorRange` object specifying the desired colors.

        Returns:

            :obj:`numpy array`: binary image with the regions of the image that fall in the color range

            :obj:`numpy array`: binary image with the edges in the image that fall in the color range
        """
        # threshold colors in HSV space
        map = color_range.inRange(self.hsv)

        # binary dilation: fills in gaps and makes the detected regions grow
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.dilation_kernel_size, self.dilation_kernel_size)
        )
        map = cv2.dilate(map, kernel)

        # extract only the edges which come from the region with the selected color
        edge_color = cv2.bitwise_and(map, self.canny_edges)

        return map, edge_color

    def findNormal(self, map, lines):
        """
        Calculates the centers of the line segments and their normals.

        Args:
            map (:obj:`numpy array`):  binary image with the regions of the image that fall in a given color range

            lines (:obj:`numpy array`): An ``Nx4`` array where each row represents a line. If no lines were detected,
            returns an empty list.

        Returns:
            :obj:`tuple`: a tuple containing:

                 * :obj:`numpy array`: An ``Nx2`` array where each row represents the center point of a line. If no lines were detected returns an empty list.

                 * :obj:`numpy array`: An ``Nx2`` array where each row represents the normal of a line. If no lines were detected returns an empty list.
        """
        normals = []
        centers = []
        if len(lines) > 0:
            length = np.sum((lines[:, 0:2] - lines[:, 2:4]) ** 2, axis=1, keepdims=True) ** 0.5
            dx = 1.0 * (lines[:, 3:4] - lines[:, 1:2]) / length
            dy = 1.0 * (lines[:, 0:1] - lines[:, 2:3]) / length

            centers = np.hstack([(lines[:, 0:1] + lines[:, 2:3]) / 2, (lines[:, 1:2] + lines[:, 3:4]) / 2])
            x3 = (centers[:, 0:1] - 3.0 * dx).astype("int")
            y3 = (centers[:, 1:2] - 3.0 * dy).astype("int")
            x4 = (centers[:, 0:1] + 3.0 * dx).astype("int")
            y4 = (centers[:, 1:2] + 3.0 * dy).astype("int")

            np.clip(x3, 0, map.shape[1] - 1, out=x3)
            np.clip(y3, 0, map.shape[0] - 1, out=y3)
            np.clip(x4, 0, map.shape[1] - 1, out=x4)
            np.clip(y4, 0, map.shape[0] - 1, out=y4)

            flag_signs = (np.logical_and(map[y3, x3] > 0, map[y4, x4] == 0)).astype("int") * 2 - 1
            normals = np.hstack([dx, dy]) * flag_signs

        return centers, normals

    def detectLines(self, color_range):
        """
        Detects the line segments in the currently set image that occur in and the edges of the regions of the image
        that are within the provided colour ranges.

        Args:
            color_range (:py:class:`ColorRange`): A :py:class:`ColorRange` object specifying the desired colors.

        Returns:
            :py:class:`Detections`: A :py:class:`Detections` object with the map of regions containing the desired colors, and the detected lines, together with their center points and normals,
        """

        map, edge_color = self.colorFilter(color_range)
        lines = self.houghLine(edge_color)
        centers, normals = self.findNormal(map, lines)
        return Detections(lines=lines, normals=normals, map=map, centers=centers)

    def detectYellowLines(self, img):
        """
        Detects the line segments in the currently set image that occur in and the edges of the regions of the image
        that are within the provided colour ranges.

        Args:
            color_range (:py:class:`ColorRange`): A :py:class:`ColorRange` object specifying the desired colors.

        Returns:
            :py:class:`Detections`: A :py:class:`Detections` object with the map of regions containing the desired colors, and the detected lines, together with their center points and normals,
        """
        print('!!')

        _, lines = self._detect_dash_line(img)
        print(lines)
        lines = np.asarray(lines)

        # lines = np.asarray([[10,10,10,50],
        #                     [20,20,20,60],
        #                     [60,20,40,60],
        #                     [40,30,20,40]
        #                     ])
        map = np.full((80, 160), 255, dtype=int)
        centers, normals = self.findNormal(map, lines)
        return Detections(lines=lines, normals=normals, map=map, centers=centers)

'''
lines =  [[86 10 86  4]] normals =  [[ 1. -0.]] map =  [[0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]] centers =  [[86.  7.]]
lines =  [[ 7 14 17 13]
 [18 12 22 12]
 [ 4 21  4 15]
 [31 10 39 10]
 [ 5 26  6 29]
 [24  1 28  3]][autobot01/stop_line_filter_node-8] killing on exit
[autobot01/lane_filter_node-7] killing on exit
[INFO] [1646920190.060340]: [/autobot01/stop_line_filter_node] Received shutdown request.
[INFO] [1646920190.067721]: [/autobot01/lane_filter_node] Received shutdown request.
 normals =  [[ 0.09950372  0.99503719]
 [-0.          1.        ]
 [ 1.         -0.        ]
 [ 0.         -1.        ]
 [-0.9486833   0.31622777]
 [-0.4472136   0.89442719]] map =  [[255 255 255 ... 255 255 255]
 [255 255 255 ... 255 255 255]
 [255 255 255 ... 255 255 255]
 ...
 [  0   0   0 ...   0   0   0]
 [  0   0   0 ...   0   0   0]
 [  0   0   0 ...   0   0   0]][autobot01/ground_projection_node-6] killing on exit
 centers =  [[12.  13.5]
 [20.  12. ]
 [ 4.  18. ]
 [35.  10. ]
 [ 5.5 27.5]
 [26.   2. ]]
[INFO] [1646920190.105539]: [/autobot01/ground_projection_node] Received shutdown request.
[autobot01/line_detector_node-5] killing on exit
lines =  [] normals =  [] map =  [[0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]] centers =  []
{'RED': <custom_line_detector.detections.Detections object at 0xa71ae538>, 'WHITE': <custom_line_detector.detections.Detections object at 0xa71ae7c0>, 'YELLOW': <custom_line_detector.detections.Detections object at 0xa71ae250>}
[INFO] [1646920190.133156]: [/autobot01/line_detector_node] Received shutdown request.
lines =  [[86 10 86  4]] normals =  [[ 1. -0.]] map =  [[0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]] centers =  [[86.  7.]]
lines =  [[35 10 41 10]
 [ 4 21  4 15]
 [10 14 15 13]
 [ 5 26  5 22]
 [ 1 55  4 62]
 [25 11 29 11]
 [22 12 24  9]] normals =  [[ 0.         -1.        ]
 [ 1.         -0.        ]
 [ 0.19611614  0.98058068]
 [ 1.         -0.        ]
 [-0.91914503  0.3939193 ]
 [-0.          1.        ]
 [ 0.83205029  0.5547002 ]] map =  [[255 255 255 ... 255 255 255]
 [255 255 255 ... 255 255 255]
 [255 255 255 ... 255 255 255]
 ...
 [  0   0   0 ...   0   0   0]
 [  0   0   0 ...   0   0   0]
 [  0   0   0 ...   0   0   0]] centers =  [[38.  10. ]
 [ 4.  18. ]
 [12.5 13.5]
 [ 5.  24. ]
 [ 2.5 58.5]
 [27.  11. ]
 [23.  10.5]]
lines =  [] normals =  [] map =  [[0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]] centers =  []
{'RED': <custom_line_detector.detections.Detections object at 0xa71ae538>, 'WHITE': <custom_line_detector.detections.Detections object at 0xa71ae778>, 'YELLOW': <custom_line_detector.detections.Detections object at 0xa71ae1a8>}

'''