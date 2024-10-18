#!/usr/bin/env python3
import numpy as np
import rclpy
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

class HeadingController(BaseHeadingController):
    def __init__(self):
        super().__init__('HeadingController')
        self.kp = 2.0
        self.get_logger().info("HeadingController Created")
    
    def compute_control_with_goal(self,state,goal):
        msgs = TurtleBotControl()
        msgs.omega = self.kp*wrap_angle(goal.theta-state.theta)
        return msgs
        
if __name__ == "__main__":
    rclpy.init()
    headingcontroller = HeadingController()
    rclpy.spin(headingcontroller)
    # headingcontroller.destroy_node()
    rclpy.shutdown()