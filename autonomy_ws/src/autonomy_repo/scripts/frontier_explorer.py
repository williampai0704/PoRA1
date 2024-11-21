#!/usr/bin/env python3
import rclpy                    # ROS2 client library
from rclpy.node import Node
from asl_tb3_msgs.msg import TurtleBotState
from asl_tb3_lib.grids import StochOccupancyGrid2D
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool
import numpy as np
from scipy.signal import convolve2d

class FrontierExplorer(Node):
    def __init__(self) -> None:
        super().__init__("frontier_explore_node") # node name
        self.occupancy = None
        self.state = TurtleBotState()
        self.frontier_states = []
        
        self.nav_success_sub = self.create_subscription(Bool,"/nav_success",self.navigator_callback,10)
        self.goal_state_pub = self.create_publisher(TurtleBotState, "/cmd_nav", 10)
        
        self.state_sub = self.create_subscription(TurtleBotState,"/state",self.turtle_state_callback,10)
        self.map_sub = self.create_subscription(OccupancyGrid,"/map",self.map_callback,10)
        
        
    def navigator_callback(self,msg:Bool) -> None:
        self.explore(self.occupancy)
        
    # callback for /state topic
    def turtle_state_callback(self, state: TurtleBotState) -> None:
        self.state = state

    # callback for /map topic
    def map_callback(self, msg: OccupancyGrid) -> None:
        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=7,
            probs=msg.data
        )
    
    def explore(self,occupancy):
        """ returns potential states to explore
        Args:
            occupancy (StochasticOccupancyGrid2D): Represents the known, unknown, occupied, and unoccupied states. See class in first section of notebook.

        Returns:
            frontier_states (np.ndarray): state-vectors in (x, y) coordinates of potential states to explore. Shape is (N, 2), where N is the number of possible states to explore.

        HINTS:
        - Function `convolve2d` may be helpful in producing the number of unknown, and number of occupied states in a window of a specified cell
        - Note the distinction between physical states and grid cells. Most operations can be done on grid cells, and converted to physical states at the end of the function with `occupancy.grid2state()`
        """

        window_size = 13    # defines the window side-length for neighborhood of cells to consider for heuristics
        ########################### Code starts here ###########################
        probs = occupancy.probs
        window = np.ones((window_size, window_size))

        unknown_mask = (probs == -1).astype(float)
        occupied_mask = (probs >= 0.5).astype(float)


        unknown_count = convolve2d(unknown_mask, window, mode='same', fillvalue = 1)
        occupied_count = convolve2d(occupied_mask, window, mode='same')
        unoccupied_count = window_size**2 - occupied_count - unknown_count

        unknown_percentage = unknown_count / (window_size**2)
        occupied_percentage = occupied_count / (window_size**2)
        unoccupied_percentage = unoccupied_count / (window_size**2)

        condition1 = unknown_percentage >= 0.2
        condition2 = occupied_count == 0
        condition3 = unoccupied_percentage >= 0.3

        x_,y_ = np.meshgrid(np.arange(occupancy.size_xy[0]), np.arange(occupancy.size_xy[1]))
        frontier_grid = np.stack([x_, y_],axis = -1)

        valid_pt = condition1 & condition2 & condition3
        if not valid_pt:
            self.get_logger().info("Finished Exploring")
            return
        Frontier = frontier_grid[valid_pt]
        self.frontier_states = occupancy.grid2state(Frontier)
        current_state = np.array([self.state.x, self.state.y])
        distances = np.linalg.norm(self.frontier_states - current_state, axis=1)
        
        min_index = np.argmin(distances)
        min_dist_state = self.frontier_states[min_index, :]
        new_goal_state = TurtleBotState()
        self.get_logger().info(f"GOAL: {min_dist_state}")
        new_goal_state.x = min_dist_state[0]
        new_goal_state.y = min_dist_state[1]
        new_goal_state.theta = 0.0 # set to 0 by default
        self.goal_state_pub.publish(new_goal_state)
        
if __name__ == "__main__":
    rclpy.init()
    node = FrontierExplorer()
    rclpy.spin(node)
    rclpy.shutdown()
