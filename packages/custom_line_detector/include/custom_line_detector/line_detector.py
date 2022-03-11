import cv2
import numpy as np

from line_detector_interface import LineDetectorInterface
from .detections import Detections


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

    def detectYellowLines(self, color_range):
        """
        Detects the line segments in the currently set image that occur in and the edges of the regions of the image
        that are within the provided colour ranges.

        Args:
            color_range (:py:class:`ColorRange`): A :py:class:`ColorRange` object specifying the desired colors.

        Returns:
            :py:class:`Detections`: A :py:class:`Detections` object with the map of regions containing the desired colors, and the detected lines, together with their center points and normals,
        """

        map = np.full((80, 160), 255, dtype=int)
        lines = np.asarray([[10,10,10,50],
                            [20,20,20,60],
                            [60,20,40,60],
                            [40,30,20,40]
                            ])

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