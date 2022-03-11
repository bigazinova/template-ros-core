"""

    custom_line_detector
    -------------

    The ``custom_line_detector`` library packages classes and tools for handling line section extraction from images. The
    main functionality is in the :py:class:`LineDetector` class. :py:class:`Detections` is the output data class for
    the results of a call to :py:class:`LineDetector`, and :py:class:`ColorRange` is used to specify the colour ranges
    in which :py:class:`LineDetector` is looking for line segments.

    There are two plotting utilities also included: :py:func:`plotMaps` and :py:func:`plotSegments`

    .. autoclass:: custom_line_detector.Detections

    .. autoclass:: custom_line_detector.ColorRange

    .. autoclass:: custom_line_detector.LineDetector

    .. autofunction:: custom_line_detector.plotMaps

    .. autofunction:: custom_line_detector.plotSegments


"""

from .line_detector import LineDetector
from .detections import Detections
from .color_range import ColorRange
from .plot_detections import plotSegments, plotMaps
