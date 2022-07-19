#!/usr/bin/env python3
import sys
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from solution import solution


class MyNode(DTROS):

    def __init__(self, node_name):
        super(MyNode, self).__init__(node_name=node_name, node_type=NodeType.DEBUG)
        self.pub = rospy.Publisher("~car_cmd", Twist2DStamped, queue_size=1)
        self.bridge = CvBridge()
        self.cur_img = None
        self.sub_image = rospy.Subscriber(
            "/autobot02/camera_node/image/compressed",
            #"~image",
            CompressedImage,
            self.action,
            queue_size=1
        )

    def run(self):
        pass

    def action(self, image_msg):
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(image_msg)
            self.cur_img = image
            
        except ValueError as e:
            self.logerr('Could not decode image: %s' % e)
            return
        vel, omega = solution(self.cur_img)
        msg = Twist2DStamped()
        msg.v = vel
        msg.omega = omega
        self.pub.publish(msg)
        sys.stdout.flush()
   

    def on_shutdown(self):
        """Shutdown procedure.
        Publishes a zero velocity command at shutdown."""
        msg = Twist2DStamped()
        msg.v = 0.0
        msg.omega = 0.0
        self.pub.publish(msg)

        super(MyNode, self).on_shutdown()

if __name__ == '__main__':
    # create the node
    node = MyNode(node_name='circle_drive_node')
    # run node
    node.run()
    # keep spinning
    rospy.spin()
