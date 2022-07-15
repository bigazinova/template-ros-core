#!/usr/bin/env python3
import sys
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

class MyNode(DTROS):

    def __init__(self, node_name):
        super(MyNode, self).__init__(node_name=node_name, node_type=NodeType.DEBUG)
        self.pub = rospy.Publisher("~car_cmd", Twist2DStamped, queue_size=1)
        self.img_sub = rospy.Subscriber("~image", CompressedImage, self.clb)
        self.bridge = CvBridge()

    def clb(self, img):
        image = self.bridge.compressed_imgmsg_to_cv2(img)
        print(image)
    
    def run(self):
        # publish message every 1 second
        rate = rospy.Rate(1) # 1Hz
        while not rospy.is_shutdown():
            msg = Twist2DStamped()
            msg.v = 0.0
            msg.omega = 1.0
            rospy.loginfo("Publishing message 0/0.5")
            self.pub.publish(msg)
            rate.sleep()
            msg.omega = 0.0
            rospy.loginfo("Publishing message 0/0.0")
            self.pub.publish(msg)
            rate.sleep()
            sys.stdout.flush()

if __name__ == '__main__':
    # create the node
    node = MyNode(node_name='circle_drive_node')
    # run node
    node.run()
    # keep spinning
    rospy.spin()
