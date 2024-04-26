import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import OccupancyGrid
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.stats
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import cv2 as cv
#from pyamaze import maze,agent,textLabel
from queue import PriorityQueue
import math
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import scipy.stats
from typing import Tuple
# from .lib.maze_manipulation import get_waypoints, dilate123
# from .lib.pid_tf2 import move_straight, move_turn, return_cur_pos
# from .lib.occupy_nodes import first_scan, a_star_scan, return_odata_origin, a_star_search, go_to_doors
# from .lib.open_door_http import open_door
# from .lib.bucket_utils import move_to_bucket
# from .lib.servo_client import launch_servo

UNKNOWN = 1
UNOCCUPIED = 2
OCCUPIED = 3

# constants
occ_bins = [-1, 0, 50, 100]
path_main = []
dilate_size = 2
global quit
quit = 0


class FirstOccupy(Node):
    def __init__(self):
        super().__init__("firstoccupy")
        self.subscription = self.create_subscription(
            OccupancyGrid, "map", self.occ_callback, qos_profile_sensor_data
        )
        self.subscription  # prevent unused variable warning
        # occdata = np.array([])
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

    def occ_callback(self, msg):
        occdata = np.array(msg.data)
        _, _, binnum = scipy.stats.binned_statistic(
            occdata, np.nan, statistic="count", bins=occ_bins
        )
        odata = np.uint8(binnum.reshape(msg.info.height, msg.info.width))

        (odata_y, odata_x) = odata.shape
        minnum = 10000000
        current_min = (0, 0)
        for j in range(odata_y):
            for i in range(odata_x):
                if odata[j][i] == 2 and i + j < minnum:
                    minnum = i + j
                    current_min = (i, j)

        global odata_origin
        odata_origin = current_min
        
        current_min = (
            current_min[0] * msg.info.resolution + msg.info.origin.position.x,
            current_min[1] * msg.info.resolution + msg.info.origin.position.y,
        )


        self.get_logger().info("New Origin: " + str(current_min))
        # self.get_logger().info(str(msg.info.resolution))
        # resolution = 0.05m per 1 array unit

        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "origin"

        t.transform.translation.x = current_min[0]
        t.transform.translation.y = current_min[1]
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_static_broadcaster.sendTransform(t)
class Occupy(Node):
    def __init__(self):
        super().__init__("occupy")
        
        self.subscription = self.create_subscription(
            OccupancyGrid, "/global_costmap/costmap", self.occ_callback, qos_profile_sensor_data
        )
        self.subscription  # prevent unused variable warning
        # occdata = np.array([])
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer, self)

    def occ_callback(self, msg):
        global map_resolution
        global origin_pos_x
        global origin_pos_y
        
        map_resolution = msg.info.resolution
        origin_pos_x = msg.info.origin.position.x
        origin_pos_y = msg.info.origin.position.y

        global costmap
        costmap = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        # the process for obtaining odata (the occupancy grid)
        occdata = np.array(msg.data)
        # _, _, binnum = scipy.stats.binned_statistic(
        #     occdata, np.nan, statistic="count", bins=occ_bins
        # )
        # make odata a global variable to be accessible by all functions
        binnum = occdata.copy()
        binnum[occdata == -1] = 1
        binnum[occdata >= 0] = 2
        binnum[occdata == 100] = 3
        global odata
        odata = np.uint8(binnum.reshape(msg.info.height, msg.info.width))

        # dilate_size = int((0.243/2)//msg.info.resolution)
        # odata = dilate123(odata, size=dilate_size)
        # Robot is 0.243m at its longest, divide by 2 to get radius, divide by resolution to change to odata coords, no need to +1 since walls aren't 5cm
        global odata_x, odata_y
        (odata_y, odata_x) = odata.shape
        # self.get_logger().info("maze dilated")

        # obtain the rviz coordinates of the turtlebot
        # usually requires the Occupy() node to spin more than once
        try:
            trans = self.tfBuffer.lookup_transform('map', 'base_link', rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().info('No transformation found')
            return
        cur_pos = trans.transform.translation 
        self.get_logger().info('Trans: %f, %f' % (cur_pos.x, cur_pos.y))

        # confirm our defined origin coordinates in odata, obtained from FirstOccupancy()
        self.get_logger().info('odata_origin: ' + str(odata_origin))

        # rviz coordinates of turtlebot's current position
        temp_cur_pos = (cur_pos.x, cur_pos.y) 
        # convert turtlebot's rviz coordinates to odata coordintates to be reflected in the occupancy grid
        curr_pos_odata = convert_to_odata(temp_cur_pos) 
        self.get_logger().info('curr_pos_odata: ' + str(curr_pos_odata))

        # current position of turtlebot in odata, with reference to origin 
        global curr_pos
        curr_pos = reference_to_origin(curr_pos_odata)
        self.get_logger().info('Current' + str(curr_pos))

       
        # goal = (0, 0)
        # # iterate through odata to find the goal (highest y coordinate)
        # maxnum = 0
        # for row in range(odata_y):
        #     for col in range(odata_x): 
        #         is_unoccupied = (odata[row][col] == 2)
        #         is_next_to_unknown = (check_neighbours(row, col, 1))
        #         # if odata[row][col] == 2 and row + col > maxnum:
        #         if row + col > maxnum and is_unoccupied and is_next_to_unknown:
        #             maxnum = row + col
        #             goal = (col, row)
        # goal_pos = reference_to_origin(goal)


            
        # find goal_pos, the goal relative to the origin coordinates
       
        
        # raise SystemExit

        # EVERYTHING SHOULD NOW BE CONVERTED TO ODATA COORDINATES
        # exit this node once start and goal is found
        raise SystemExit
def move_straight(
    target: Tuple[float, float],
    end_distance_range: float = 0.1,
    PID_angular: Tuple[float, float, float] = (0.5, 0, 1),
    PID_linear: Tuple[float, float, float] = (0.3, 0, 1),
    angular_speed_limit: float = 1,  # old 2.84
    linear_speed_limit: float = 0.2,  # old 0.22
):
    """Move Straight to RViz Waypoint

    Args:
        target (Tuple[float, float]): (x,y) in RViz coordinates
        end_distance_range (float, optional): When target is x meters away, stop function. Defaults to 0.1.
        PID_angular (Tuple[float, float, float], optional): (kP, kI, kD). Defaults to (0.5, 0, 1).
        PID_linear (Tuple[float, float, float], optional): (kP, kI, kD). Defaults to (0.3, 0, 1).
        angular_speed_limit (float, optional): Angular Velocity Limit. Defaults to 1.
        linear_speed_limit (float, optional): Linear Velocity Limit. Defaults to 0.1.
    """
    wpmover = WPMover(
        target,
        end_distance_range,
        PID_angular,
        PID_linear,
        angular_speed_limit,
        linear_speed_limit,
    )
    try:
        rclpy.spin(wpmover)
    except Exception and KeyboardInterrupt:
        stop_kill(wpmover)
        print("killed")
    except SystemExit:
        print("sys exit done")

def return_cur_pos():
    return cur_pos
    
def move_turn(
    target: Tuple[float, float],
    end_yaw_range: float = 0.05,
    PID_angular: Tuple[float, float, float] = (1, 0, 2),
    angular_speed_limit: float = 2,  # old 2.84
):
    """Turn to face RViz Waypoint

    Args:
        target (Tuple[float, float]): (x,y) in RViz coordinates
        end_yaw_range (float, optional): When robot is within x rad, stop function. Defaults to 0.05.
        PID_angular (Tuple[float, float, float], optional): (kP, kI, kD). Defaults to (1, 0, 2).
        angular_speed_limit (float, optional): Angular Velocity Limit. Defaults to 1.
    """
    wpturner = WPTurner(target, end_yaw_range, PID_angular, angular_speed_limit)
    try:
        rclpy.spin(wpturner)
    except Exception and KeyboardInterrupt:
        stop_kill(wpturner)
        print("killed")
    except SystemExit:
        print("sys exit done")
def move_to_bucket(threshold=0.04, dist=0.33, iter=5, bucket_radius=0.135):



    """Move to bucket based on LiDAR

    Args:
        threshold (float (meters), optional): Threshold for bucket detection. Defaults to 0.04.
        dist (float (meters), optional): Distance before bucket to stop (measured to center of bucket). Defaults to 0.33.
        iter (int, optional): Number of times to collect and average bucket location. Defaults to 5.
        bucket_radius (float (meters), optional): Radius of bucket. Defaults to 0.135.
    """
    bucket_scanner = BucketScanner(threshold=threshold, bucket_radius=bucket_radius)
    # print(bucket_scanner.run_check())  # On actual robot, angle_increment always changes
    if bucket_scanner.pub_bucket(iter=iter) is not None:
        bucket_scanner.move_to_bucket(dist=dist)
    bucket_scanner.destroy_node()

# I got this one from online somewhere
def a_star_search(graph, start, goal):
    """ This function searches for the shortest path via the a star search algo
    Args: 
        graph (2D Array): A matrix representation of the map
        start (tuple): The start position of the robot
        goal (tuple): The target or goal position 
    Returns: 
        came_from (dict): For each point in the path, the coordinates provided is the coordinates prior to that point
        cost_so_far (dict): 
        final_pos (tuple): The coordinates of the final position of the robot once the path is complete"""
    print("Start: " + str(start))
    print("Goal: " + str(goal))
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    turning_cost = 5
    final_pos = 0 # initialise final_pos variable, if 0 is returned then a clear path is not found

    while not frontier.empty():
        (_,current) = frontier.get()
        # print("Current: " + str(current))
        # path.append(current)

        # bool variables to check if the current position is within range of goal
        range_dist = dilate_size # if we dilate by 2, this needs to be 2
        is_within_goal_y = (current[1] < goal[1] + range_dist and current[1] > goal[1] - range_dist)
        is_within_goal_x = (current[0] < goal[0] + range_dist and current[0] > goal[0] - range_dist)

        if is_within_goal_x and is_within_goal_y:
        # if current == goal:
            print("Goal Found!!!")
            final_pos = current
            print(final_pos)
            break

        # if current == goal:
        #     break
        
        for next in neighbors(current, graph):
            new_cost = cost_so_far[current] + cost_to_goal(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                priority += costmap[current[1]][current[0]]
                prev = came_from[current]
                if prev != None:
                    # next_direction = (int(next[0] - current[0]), int(next[1] - current[1]))
                    # current_direction = (int(current[0] - prev[0]), int(current[1] - prev[1]))
                    next_direction = (next[0] - current[0], next[1] - current[1])
                    current_direction = (current[0] - prev[0], current[1] - prev[1])
                    if  current_direction != next_direction:
                        
                        priority += turning_cost
                        # print("cost added")
                    
                frontier.put((priority,next))
                came_from[next] = current
    # print(cost_so_far)
   
    return came_from, cost_so_far, final_pos
    # return path

def first_scan(): 
    firstoccupy = FirstOccupy()

    rclpy.spin_once(firstoccupy)
    firstoccupy.destroy_node()

def a_star_scan(): 
    
    occupy = Occupy()
    # costmapsub = CostmapSub()
    # rclpy.spin_once(costmapsub)
    try:
        rclpy.spin(occupy)
    except SystemExit:                 # <--- process the exception 
        rclpy.logging.get_logger("Quitting").info('Done')
        
    occupy.destroy_node()

    goal_pos = get_goal(odata_x - 1, odata_y - 1)
    print("Goal: " + str(goal_pos))

    path_main = 0
    while path_main == 0:
        path_main = get_path(curr_pos, goal_pos)
        goal_pos = get_goal(goal_pos[0], goal_pos[1]) # start finding new goal from the next row onwards, iterate from goal_pos[1] and below

    return path_main

def go_to_doors(goal=(1.8, 2.8)):
    print("Going to doors!!!")
    occupy = Occupy()
    # costmapsub = CostmapSub()
    # rclpy.spin_once(costmapsub)
    try:
        rclpy.spin(occupy)
    except SystemExit:                 # <--- process the exception 
        rclpy.logging.get_logger("Quitting").info('Done')
    
    occupy.destroy_node()
    # path_main = get_path(curr_pos, goal_pos)
     # x = 1.7, y = 2.9
    goal_rviz = goal
    print("goal_rviz: " + str(goal_rviz))
    goal_odata = reference_to_origin(reference_to_origin(convert_to_odata(goal_rviz)))
    print("goal_odata" + str(goal_odata))
    #     print("going between doors")
    #     goal_pos =
    path_main = get_path(curr_pos, goal_odata)
    print(str(path_main))
    return path_main

def return_odata_origin():
    return odata_origin

def get_waypoints(path_array:list): # path_array in rviz coord
    """Generate waypoints (vertices) from shortest path

    Args:
        path_array (list): List of each individual point (in (x,y) rviz coordinates) of the shortest path

    Returns:
        waypoints_array (list): List of waypoints (in rviz coordinates format)
    """
    waypoints_array = []
    prev_diff = (round((path_array[1][0] - path_array[0][0]),2), round((path_array[1][1] - path_array[0][1]),2))
    for i in range(1,len(path_array)):
        current_diff = (round((path_array[i][0] - path_array[i-1][0]),2), round((path_array[i][1] - path_array[i-1][1]),2))
        if prev_diff != current_diff:
            prev_diff = current_diff
            print(prev_diff)
            print(current_diff)
            waypoints_array.append(path_array[i-1])
    waypoints_array.append(path_array[-1])
    return waypoints_array

def dilate123(src, size=1, shape=cv.MORPH_RECT):
    """Dilation for odata array (after binning and reshape)

    Args:
        src (ndarray): input odata array
        size (integer): number of iterations of dilation,
                            this size / resolution is the meters that the image is dilated.
        shape (integer): cv2 MorphShapes

    Returns:
        ndarray: output of dilated array
    """
    array_edited = np.copy(src)
    array_edited[array_edited <= 2] = 0
    array_edited //= 3
    array_dilated = cv.dilate(
        array_edited,
        cv.getStructuringElement(shape, (2 * size + 1, 2 * size + 1)),
    )
    array_dilated *= 3
    return np.maximum(src, array_dilated)
class mapCheck(Node):
    def __init__(self):
        super().__init__("mapcheck")
        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/global_costmap/costmap',
            self.occ_callback,
            10)
        self.subscription  # prevent unused variable warning
        # occdata = np.array([])
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer, self)
    
    def occ_callback(self, msg):
        occdata = np.array(msg.data)
        cdata = occdata.reshape(msg.info.height, msg.info.width)
        cdata[cdata == 100] = -1
        cdata[cdata >= 0] = 1
        cdata[cdata == -1] = 0
        no_wall_indexes = np.nonzero(cdata)
        y_dist = np.max(no_wall_indexes[0]) - return_odata_origin()[1]
        print("distance_to_furthest:  "+str(y_dist))
        if (y_dist > (2.85 / msg.info.resolution)):
            global quit
            quit = 1
            print("!!!!!!!!!!!!!!!!!!!!!!!!quit!!!!!!!!!!!!!!!!!!")
        

def main(args=None):
    rclpy.init(args=args)

    first_scan()

    # create matplotlib figure
    plt.ion()
    plt.show()
    mapcheck = mapCheck()


    for _ in range(15):
        path_main = a_star_scan()

        outwps = get_waypoints(path_main)
        print("out waypoints: " + str(outwps))
        time_start = time.time()
        for x in outwps:
            print(x)
            # time.sleep(2)
            if quit:
                break
            if time.time()-time_start > 20:
                break
            # will reset once every 20 seconds unless exit is seen: if exit seen, will move directly to exit and skip the resets.
            # once exit is seen, don't reset anymore (exitbreak will never equal 1) until quit is called
            move_turn(x, end_yaw_range=0.13, PID_angular=(2,0,4))
            # time.sleep(1)
            move_straight(x)
            # time.sleep(1)

            rclpy.spin_once(mapcheck)
        if quit:
            print("quit, at maze exit")
            break

    path_main = go_to_doors()
    outwps = get_waypoints(path_main)
    print("out waypoints: " + str(outwps))
    for x in outwps:
        print(x)
        # time.sleep(2)
        move_turn(x, end_yaw_range=0.13, PID_angular=(2,0,4))
        # time.sleep(1)
        move_straight(x)
        # time.sleep(1)

        rclpy.spin_once(mapcheck)
    

    # door = open_door("192.168.67.")
    # TODO move to either room 1 or 2
    # move_to_bucket()
    # launch_servo()

    # Go back to explore the maze
    for _ in range(15):
        path_main = a_star_scan()

        outwps = get_waypoints(path_main)
        print("out waypoints: " + str(outwps))
        time_start = time.time()
        for x in outwps:
            print(x)
            # time.sleep(2)
            # will reset once every 20 seconds unless exit is seen: if exit seen, will move directly to exit and skip the resets.
            # once exit is seen, don't reset anymore (exitbreak will never equal 1) until quit is called
            move_turn(x, end_yaw_range=0.13, PID_angular=(2,0,4))
            # time.sleep(1)
            move_straight(x)
            # time.sleep(1)

            rclpy.spin_once(mapcheck)
    
    plt.close()


    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    # occupy.destroy_node()
    rclpy.shutdown()
    # cv.destroyAllWindows()


if __name__ == "__main__":
    main()
