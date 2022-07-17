#!/usr/bin/env python3
import sys
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped

class MyNode(DTROS):

    def __init__(self, node_name):
        super(MyNode, self).__init__(node_name=node_name, node_type=NodeType.DEBUG)
        self.pub = rospy.Publisher("~car_cmd", Twist2DStamped, queue_size=1)

    def run(self):
        pass
            
    def on_shutdown(self):
        pass

if __name__ == '__main__':
    # create the node
    node = MyNode(node_name='circle_drive_node')
    # run node
    node.run()
    # keep spinning
    rospy.spin()
