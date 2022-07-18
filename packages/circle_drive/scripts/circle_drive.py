#!/usr/bin/env python3
import sys
import rospy
from time import sleep
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped # v w

class MyNode(DTROS):

    def __init__(self, node_name):
        super(MyNode, self).__init__(node_name=node_name, node_type=NodeType.DEBUG)
        self.pub = rospy.Publisher("~car_cmd", Twist2DStamped, queue_size=1)

    def run(self):
        msg = Twist2DStamped()
        # rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            msg.omega = 1
            self.pub.publish(msg)
            sleep(1)
            msg.omega = 0
            self.pub.publish(msg)
            #sleep(1)

            
            
            
    def on_shutdown(self):
        msg = Twist2DStamped()
        msg.v = 0
        msg.omega = 0
        self.pub.publish(msg)

if __name__ == '__main__':
    # create the node
    node = MyNode(node_name='circle_drive_node')
    # run node
    node.run()
    # keep spinning
    rospy.spin()
