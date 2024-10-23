#!/usr/bin/env python3
import rclpy                    # ROS2 client library
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.tf_utils import quaternion_to_yaw
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from astar import AStar
from asl_tb3_lib.grids import StochOccupancyGrid2D
import scipy.interpolate 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class Navigator(BaseNavigator):
    def __init__(self) -> None:
        # give it a default node name
        super().__init__("navigator")
        self.kp = 2.0
        self.kpx = 2.0
        self.kpy = 2.0
        self.kdx = 2.0
        self.kdy = 2.0
        self.V_PREV_THRES = 0.0001
        self.t_prev = 0.
        self.V_prev = 0.
        self.v_desired=0.15
        self.spline_alpha=0.05
        self.get_logger().info("Navigator Created")
    
    def compute_heading_control(self, state, goal):
        msgs = TurtleBotControl()
        msgs.omega = self.kp*wrap_angle(goal.theta-state.theta)
        return msgs
    
    def compute_trajectory_tracking_control(self, state, plan, t):
        x = state.x
        y = state.y
        th = state.theta
        dt = t-self.t_prev
        x_d = scipy.interpolate.splev(t, plan.path_x_spline, der=0)
        y_d = scipy.interpolate.splev(t, plan.path_y_spline, der=0)

        # Velocity
        xd_d = scipy.interpolate.splev(t, plan.path_x_spline, der=1)
        yd_d = scipy.interpolate.splev(t, plan.path_y_spline, der=1)

        # Acceleration
        xdd_d = scipy.interpolate.splev(t, plan.path_x_spline, der=2)
        ydd_d = scipy.interpolate.splev(t, plan.path_y_spline, der=2)
        u1 = xdd_d + self.kpx*(x_d-x) + self.kdx*(xd_d-self.V_prev*np.cos(th))
        u2 = ydd_d + self.kpy*(y_d-y) + self.kdy*(yd_d-self.V_prev*np.sin(th))
        V = self.V_prev + (u1*np.cos(th)+u2*np.sin(th))*dt
        if(V <= self.V_PREV_THRES):
            V = self.V_PREV_THRES
        om = (-u1*np.sin(th)+u2*np.cos(th))/V
        
        control = TurtleBotControl()
        control.v = V
        control.omega = om
        return control
    
    def compute_trajectory_plan(self, state: TurtleBotState, goal: TurtleBotState,
                                occupancy: StochOccupancyGrid2D, resolution: float, 
                                horizon: float) -> TrajectoryPlan | None:
        x_init = np.array([state.x,state.y])
        x_goal = np.array([goal.x,goal.y])
        
        astar = AStar(x_init = x_init,x_goal = x_goal,
                      statespace_lo = (x_init[0] - horizon,x_init[1] - horizon),
                      statespace_hi=(x_init[0] + horizon,x_init[1] + horizon),
                      occupancy=occupancy,resolution=resolution)
        if not astar.solve() or len(astar.path) < 4:
            print("No path found")
            self.get_logger().info("Path Not Found")
            return None
        # Reset
        self.V_prev = 0.
        self.t_prev = 0.
        
        path = np.asarray(astar.path)
        ts = np.cumsum(np.linalg.norm(np.diff(path,axis = 0),axis = -1)/self.v_desired)
        ts = np.insert(ts,0,0.0)
        path_x_spline = scipy.interpolate.splrep(ts,path[:,0],s = self.spline_alpha)
        path_y_spline = scipy.interpolate.splrep(ts,path[:,1],s = self.spline_alpha)
        
        return TrajectoryPlan(
        path=path,
        path_x_spline=path_x_spline,
        path_y_spline=path_y_spline,
        duration=ts[-1],
    )    

if __name__ == "__main__":
    rclpy.init()            # initialize ROS client library
    node = Navigator()    # create the node instance
    rclpy.spin(node)        # call ROS2 default scheduler
    rclpy.shutdown()        # clean up after node exits