#!/usr/bin/env python3
import rclpy                    # ROS2 client library
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.tf_utils import quaternion_to_yaw
from asl_tb3_msgs.msg import TurtleBotControl
import scipy.interpolate 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_offset = x_init                     
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0.0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
              useful here
        """
        ########## Code starts here ##########
        return self.occupancy.is_free(x)
        ########## Code ends here ##########

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        HINT: This should take one line. Tuples can be converted to numpy arrays using np.array().
        """
        ########## Code starts here ##########
        return np.linalg.norm(np.array(x1)-np.array(x2))
        ########## Code ends here ##########

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by adding/subtracting self.resolution from x,
               numerical errors could creep in over the course of many additions
               and cause grid point equality checks to fail. To remedy this, you
               should make sure that every neighbor is snapped to the grid as it
               is computed.
        """
        neighbors = []
        ########## Code starts here ##########
        if(self.is_free(self.snap_to_grid((x[0]+self.resolution,x[1])))):
            neighbors.append(self.snap_to_grid((x[0]+self.resolution,x[1])))
        if(self.is_free(self.snap_to_grid((x[0]-self.resolution,x[1])))):
            neighbors.append(self.snap_to_grid((x[0]-self.resolution,x[1])))
        if(self.is_free(self.snap_to_grid((x[0],x[1]+self.resolution)))):
            neighbors.append(self.snap_to_grid((x[0],x[1]+self.resolution)))
        if(self.is_free(self.snap_to_grid((x[0],x[1]-self.resolution)))):
            neighbors.append(self.snap_to_grid((x[0],x[1]-self.resolution)))
        if(self.is_free(self.snap_to_grid((x[0]+self.resolution/(2**(1/2)),x[1]+self.resolution/(2**(1/2)))))):
            neighbors.append(self.snap_to_grid((x[0]+self.resolution/(2**(1/2)),x[1]+self.resolution/(2**(1/2)))))
        if(self.is_free(self.snap_to_grid((x[0]+self.resolution/(2**(1/2)),x[1]-self.resolution/(2**(1/2)))))):
            neighbors.append(self.snap_to_grid((x[0]+self.resolution/(2**(1/2)),x[1]-self.resolution/(2**(1/2)))))
        if(self.is_free(self.snap_to_grid((x[0]-self.resolution/(2**(1/2)),x[1]+self.resolution/(2**(1/2)))))):
            neighbors.append(self.snap_to_grid((x[0]-self.resolution/(2**(1/2)),x[1]+self.resolution/(2**(1/2)))))
        if(self.is_free(self.snap_to_grid((x[0]-self.resolution/(2**(1/2)),x[1]-self.resolution/(2**(1/2)))))):
            neighbors.append(self.snap_to_grid((x[0]-self.resolution/(2**(1/2)),x[1]-self.resolution/(2**(1/2)))))
        ########## Code ends here ##########
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))
    '''
    def plot_path(self, fig_num=0, show_init_label=True):
        """Plots the path found in self.path and the obstacles"""
        if not self.path:
            return

        self.occupancy.plot(fig_num)

        solution_path = np.asarray(self.path)
        plt.plot(solution_path[:,0],solution_path[:,1], color="green", linewidth=2, label="A* solution path", zorder=10)
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        if show_init_label:
            plt.annotate(r"$x_{init}$", np.array(self.x_init) + np.array([.2, .2]), fontsize=16)
        plt.annotate(r"$x_{goal}$", np.array(self.x_goal) + np.array([.2, .2]), fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

        plt.axis([0, self.occupancy.width, 0, self.occupancy.height])

    def plot_tree(self, point_size=15):
        plot_line_segments([(x, self.came_from[x]) for x in self.open_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        plot_line_segments([(x, self.came_from[x]) for x in self.closed_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        px = [x[0] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        py = [x[1] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        plt.scatter(px, py, color="blue", s=point_size, zorder=10, alpha=0.2)
    '''
    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found

        HINTS:  We're representing the open and closed sets using python's built-in
                set() class. This allows easily adding and removing items using
                .add(item) and .remove(item) respectively, as well as checking for
                set membership efficiently using the syntax "if item in set".
        """
        ########## Code starts here ##########
        while self.open_set:
            x_curr = self.find_best_est_cost_through()
            if x_curr == self.x_goal:
                self.path = self.reconstruct_path()
                return True
            self.open_set.remove(x_curr)
            self.closed_set.add(x_curr)
            for x in self.get_neighbors(x_curr):
                if x in self.closed_set:
                    continue
                tent_cost_to_arrive  = self.cost_to_arrive[x_curr] + self.distance(x,x_curr)
                if x not in self.open_set:
                    self.open_set.add(x)
                elif tent_cost_to_arrive > self.cost_to_arrive[x]:
                    continue
                self.came_from[x] = x_curr
                self.cost_to_arrive[x] = tent_cost_to_arrive
                self.est_cost_through[x] = tent_cost_to_arrive + self.distance(x,self.x_goal)
        
        return False
        ########## Code ends here ##########

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
    
    def compute_trajectory_plan(self, state, goal, occupancy, resolution, horizon):
        astar = AStar((0,0),(horizon,horizon),(state.x,state.y),(goal.x,goal.y),occupancy=occupancy,resolution=resolution)
        if not astar.solve() or len(astar.path) < 4:
            print("No path found")
            return None
        # Reset
        self.V_prev = 0.
        self.t_prev = 0.
        
        path = np.asarray(astar.path)
        ts = np.cumsum(np.linalg.norm(np.diff(path,axis = 0),axis = -1)/v_desired)
        ts = np.insert(ts,0,0.0)
        path_x_spline = scipy.interpolate.splrep(ts,path[:,0],s = spline_alpha)
        path_y_spline = scipy.interpolate.splrep(ts,path[:,1],s = spline_alpha)
        
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