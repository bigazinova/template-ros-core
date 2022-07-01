import functools

import cv2
from math import sqrt
from line_detector_interface import LineDetectorInterface
from .detections import Detections
import numpy as np


BLACK_PIXEL = [0, 0, 0]
EDGES_NUM_IN_RECT = 4


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

    class Contour:
        def __init__(self, center, rad, approx):
            self.center = center
            self.rad = int(rad)
            self.approx = approx

        def __lt__(self, other):
            return self.rad < other.rad

        def __eq__(self, other):
            return self.center == other.center and self.rad == other.rad

        def __gt__(self, other):
            return self.rad > other.rad

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
        self.last_correct_edge = []

    @staticmethod
    def _check_cos(cos_a, cos_b, inv=False):
        DEGREES_DIST = 20
        RIGHT_ANGLE_DIST = 5
        RIGHT_ANGLE = 90
        degrees_a, degrees_b = LineDetector._get_degrees_from_cos(cos_a), LineDetector._get_degrees_from_cos(cos_b)
        if abs(degrees_a - degrees_b) > DEGREES_DIST:
            if not inv:
                return 0 if cos_a > cos_b else 1
            else:
                return 1 if cos_a > cos_b else 0
        else:
            if abs(degrees_a - RIGHT_ANGLE) <= RIGHT_ANGLE_DIST:
                if not inv:
                    return 0
                return 1
            if abs(degrees_b - RIGHT_ANGLE) <= RIGHT_ANGLE_DIST:
                if not inv:
                    return 1
                return 0
            return -1

    @staticmethod
    def _get_polynomial():
        x = np.array([74, 69, 65, 48, 45, 40, 29, 0])
        y = np.array([36, 35, 28, 27, 26, 25, 11, 0])
        y = y + 5
        return np.poly1d(np.polyfit(x, y, 4))

    @staticmethod
    def _get_max_dist_between_elements(y):
        min_dist_lim = 5
        pol = LineDetector._get_polynomial()
        result = round(pol(y))
        # result = y / 2
        return 0 if result <= min_dist_lim else result

    @staticmethod
    def _dist_pixels(pix1, pix2):
        return sqrt((pix1[0] - pix2[0]) ** 2 + (pix1[1] - pix2[1]) ** 2)

    def _filter_contours(self, contours, img):
        filtered_contours = []
        for k, contour in enumerate(contours):
            approx = cv2.approxPolyDP(contour, 0.045 * cv2.arcLength(contour, True), True)
            center = sum(approx) / approx.shape[0]
            radius = max(contour, key=lambda x: abs(center[0][0] - x[0][0]) ** 2 + abs(center[0][1] - x[0][1]) ** 2)
            radius_length = sqrt((radius[0][0] - center[0][0]) ** 2 + (radius[0][1] - center[0][1]) ** 2)

            if len(approx) == EDGES_NUM_IN_RECT:
                diagonal = self._get_diagonal(approx)
                cont_center = [int((diagonal[0][0] + diagonal[1][0]) // 2), int((diagonal[0][1] + diagonal[1][1]) // 2)]

                if np.array_equal(img[int(center[0][1]), int(center[0][0])],  BLACK_PIXEL):
                    continue

                filtered_contours.append(self.Contour(cont_center, radius_length, approx))
        return filtered_contours, img

    def _get_diagonal(self, approx):
        diagonal = []
        max_dist = 0
        for cortex_1 in approx:
            for cortex_2 in approx:
                if self._dist_pixels(cortex_1[0], cortex_2[0]) > max_dist:
                    max_dist = self._dist_pixels(cortex_1[0], cortex_2[0])
                    diagonal = [cortex_1[0], cortex_2[0]]
        return diagonal

    def get_edges_from_contours(self, img, contours):
        self.last_correct_edge = []
        edges = []
        for contour in contours:
            contour, distances = self._create_edges_from_approx(contour.approx)
            max_edge_ind = self._get_max_edge_index(contour, distances, img.shape[0] - 1)

            edge = contour[max_edge_ind]
            edges += LineDetector._get_contour_by_piece(edge)

            max_edge_ind = (max_edge_ind + 2) % EDGES_NUM_IN_RECT
            edge = contour[max_edge_ind]
            edges += LineDetector._get_contour_by_piece(edge)
        return edges

    def _get_max_edge_index(self, contour, distances, y):

        if self.last_correct_edge:
            has_last = False
            last = self.last_correct_edge
        else:
            has_last = True
            x = contour[0][0] if contour[0][0] != 0 else 10
            last = (0, y, x, y)
        if not (contour[0][0] == 0 and contour[0][2] == 0 or contour[1][0] == 0 and contour[1][2] == 0):

            cos_a = LineDetector._get_cos(last, contour[0], distances[0])
            cos_b = LineDetector._get_cos(last, contour[1], distances[1])
            edge_index = LineDetector._check_cos(cos_a, cos_b, has_last)
            if edge_index != -1:
                self.last_correct_edge = contour[edge_index]
                return edge_index

        max_dist_index = np.argmax(distances)
        if abs(distances[0] - distances[1]) > 2 and abs(distances[2] - distances[3]) > 2:
            if distances[max_dist_index] > 6:
                self.last_correct_edge = contour[max_dist_index]
                return max_dist_index
        cos_a = LineDetector._get_cos(last, contour[2], distances[2])
        cos_b = LineDetector._get_cos(last, contour[3], distances[3])

        edge_index = LineDetector._check_cos(cos_a, cos_b, has_last)
        if edge_index != -1:
            # print('by third cos')
            self.last_correct_edge = contour[edge_index]
            return edge_index
        # print('by random')
        if not has_last:
            edge_index = 0 if cos_a > cos_b else 1
        else:
            edge_index = 1 if cos_a > cos_b else 0
        self.last_correct_edge = contour[edge_index]
        return edge_index

    @staticmethod
    def _get_degrees_from_cos(cos):
        return np.degrees(np.arccos(cos))

    @staticmethod
    def _get_cos(last, current, dist_current):
        dist_last = LineDetector._dist_pixels((last[0], last[1]), (last[2], last[3]))
        last = [last[0] - last[2], last[1] - last[3]]
        current = [current[0] - current[2], current[1] - current[3]]
        try:
            return abs((last[0] * current[0] + last[1] * current[1]) / (dist_last * dist_current))
        except ZeroDivisionError:
            return 1

    @staticmethod
    def _get_contour_by_piece(edge):
        if abs(edge[0] - edge[2]) <= 1 or abs(edge[1] - edge[3]) <= 1:
            return [edge]
        x_list = list(map(int, np.linspace(edge[0], edge[2], 3)))
        y_list = list(map(int, np.linspace(edge[1], edge[3], 3)))
        contour = []
        current_piece = []
        for x, y in zip(x_list, y_list):
            if current_piece:
                current_piece.extend([x, y])
                contour.append(current_piece)
            current_piece = [x, y]
        return contour

    @staticmethod
    def _create_edges_from_approx(approx):
        vertex_0 = approx[0]
        x_0 = int(vertex_0[0][0])
        y_0 = int(vertex_0[0][1])

        current_vertex = [x_0, y_0]
        contour = []
        distances = []

        for vertex in approx[1:]:
            x_1 = int(vertex[0][0])
            y_1 = int(vertex[0][1])

            current_vertex.extend([x_1, y_1])
            distances.append(LineDetector._dist_pixels(current_vertex[:2], current_vertex[2:]))
            contour.append(current_vertex)
            current_vertex = [x_1, y_1]

        current_vertex.extend([x_0, y_0])
        contour.append(current_vertex)
        distances.append(LineDetector._dist_pixels(current_vertex[:2], current_vertex[2:]))
        return contour, distances

    @staticmethod
    def _sort_contours(contours):
        return sorted(contours, key=functools.cmp_to_key(
            lambda contour1, contour2: LineDetector._dist_pixels(contour1.center, contour2.center)), reverse=True)

    def detect_dash_line(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, threshold_image = cv2.threshold(gray, 0, 255,
                                             cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, h = cv2.findContours(threshold_image, 1, 2)
        threshold_image = cv2.cvtColor(threshold_image, cv2.COLOR_GRAY2RGB)
        contours, threshold_image = self._filter_contours(contours, threshold_image)
        if not len(contours):
            return threshold_image, []
        contours = LineDetector._sort_contours(contours)

        next_contour = False
        dash_lines = []
        prev_contour = contours[0]

        for i, contour in enumerate(contours[1:]):

            max_dist = LineDetector._get_max_dist_between_elements(prev_contour.center[1])
            dist = LineDetector._dist_pixels(contour.center, prev_contour.center)
            if dist <= max_dist and not LineDetector._is_inside(contour, prev_contour):
                dash_lines.append(prev_contour)
                next_contour = True
            else:
                if next_contour:
                    dash_lines.append(prev_contour)
                next_contour = False

            prev_contour = contour

        if next_contour:
            dash_lines.append(prev_contour)
        return threshold_image, dash_lines

    @staticmethod
    def _is_inside(first_line, second_line):
        if first_line == second_line:
            return False
        big_cont = first_line if first_line > second_line else second_line
        small_cont = first_line if first_line < second_line else second_line
        dist_between_cent = LineDetector._dist_pixels(first_line.center, second_line.center)
        if small_cont.rad + dist_between_cent < big_cont.rad:
            return True
        return False

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

    def detect_yellow_lines(self, img):
        """
        Detects the line segments in the currently set image that occur in and the edges of the regions of the image
        that are within the provided colour ranges.

        Args:
            img (:py:class:`numpy.ndarray`): A :py:class:`numpy.ndarray` object specifying the image from camera.

        Returns:
            :py:class:`Detections`: A :py:class:`Detections` object with the map of regions containing the desired colors, and the detected lines, together with their center points and normals,
        """
        map_ = np.full(img.shape[:-1], 255, dtype=int)
        debug_threshold_image, contours = self.detect_dash_line(img)
        edges = self.get_edges_from_contours(img, contours)

        if not edges:
            return Detections(lines=edges, normals=[], map=map_, centers=[])

        edges = np.asarray(edges)
        centers, normals = self.findNormal(map_, edges)
        return Detections(lines=edges, normals=normals, map=map_, centers=centers)
