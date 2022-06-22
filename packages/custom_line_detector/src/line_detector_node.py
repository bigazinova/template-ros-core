#!/usr/bin/env python3

import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from duckietown_msgs.msg import Segment, SegmentList, AntiInstagramThresholds
from custom_line_detector import LineDetector, ColorRange, plotSegments, plotMaps
from image_processing.anti_instagram import AntiInstagram
import os
from image_geometry import PinholeCameraModel
from duckietown.dtros import DTROS, NodeType, TopicType
# from camera_info_manager import CameraInfoManager

class LineDetectorNode(DTROS):

    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(LineDetectorNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self.bridge = CvBridge()

        # Publishers
        self.pub_lines = rospy.Publisher(
            "~custom/compressed", CompressedImage, queue_size=1, dt_topic_type=TopicType.DEBUG
        )
        # Subscribers
        self.sub_image = rospy.Subscriber(
            "~image/compressed", CompressedImage, self.image_cb, buff_size=1, queue_size=1
        )


    def image_cb(self, image_msg):

        # Decode from compressed image with OpenCV
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(image_msg)
        except ValueError as e:
            self.logerr(f"Could not decode image: {e}")
            return
        
        self.pub_lines.publish(self.bridge.cv2_to_compressed_imgmsg(image))


if __name__ == "__main__":
    # Initialize the node
    line_detector_node = LineDetectorNode(node_name="line_detector_node")
    # Keep it spinning to keep the node alive
    rospy.spin()
