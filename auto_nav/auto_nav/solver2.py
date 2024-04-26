import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import math
import cmath
import time
from rclpy.qos import ReliabilityPolicy, QoSProfile
import scipy.stats
import requests
import RPi.GPIO as GPIO
#import sys
#import cv2

#rpi's lidar package, fullrun.py, shortcut is turtle

startMove = 5
rotatechange = 0.35
speedchange = 0.15
stop_distance = 0.3
search_distance = 0.6

#angles will be scaled according to the length of the laser array not 360 (200+) -- adjust accordingly
front_angle = 12
front_angles = range(-front_angle,front_angle+1,1)
search_angle = 60
search_angles = range(-search_angle,search_angle+1,3)
box_thres = 0.1 #so that it reach frontier behind wall?

door_x = 1.6 #1.62
door_y = 2.70
width_map = 3.55  #2.48 + 1.46
height_map = 2.6   #1.6+1.08 #measured the table length, 1.6m
map_percent = 0.67 #!!!start with



# !!!!!RMB TO CHANGE this to the IP of your ESP32!!!!!
esp32_ip = '192.168.38.87'
#esp32_ip = '192.168.227.169'  
#esp32_ip = '192.168.38.169' 
TurtleBot3_ID = 'turtlebot'
door_time = 5   #time to go through the door before stopping (assume no obs)
dist_threshold = 0.16 # Distance threshold for the robot to stop in front of the pail


free_dis_thres = 0.2
free_dis = 0.4
long_dis = 1.5

occ_bins = [-1, 0, 50, 100]
visited = set()

#SERVO
GPIO.setmode(GPIO.BCM)##rpi , lidar, bucket
servo_pin = 13
GPIO.setup(servo_pin, GPIO.OUT)
pwm = GPIO.PWM(servo_pin, 50)
pwm.start(7.5) #90
time.sleep(0.5)

def launch_ball():
    print("Launching! Big Balls")
    duty_cycle = 2.5 + float(180) / 18#rotate 90
    pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)  # Give time for servo to move

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z # in radians

def calculate_yaw_and_distance(x1, y1, x2, y2, current_yaw):
    # Calculate the angle between the two points using the arctan2 function
    delta_x = x2 - x1
    delta_y = y2 - y1
    target_yaw = math.atan2(delta_y, delta_x)

    # Calculate the distance between the two points using the Pythagorean theorem
    distance = math.sqrt(delta_x ** 2 + delta_y ** 2)

    # Calculate the difference between the current yaw and the target yaw
    # print("target_yaw = ", target_yaw)
    # print("current_yaw = ", current_yaw)
    yaw_difference = target_yaw - current_yaw

    # Normalize the yaw difference to between -pi and pi radians
    if yaw_difference > math.pi:
        yaw_difference -= 2 * math.pi
    elif yaw_difference < -math.pi:
        yaw_difference += 2 * math.pi

    return (round(yaw_difference, 3), round(distance, 3))

class Solver(Node):
    def __init__(self):
        super().__init__('solver')
        # create publisher for moving TurtleBot
        self.publisher_ = self.create_publisher(Twist,'cmd_vel',10)
        
        # odometry 
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.pos_x = 0.0
        self.pos_y = 0.0
    
        # map grid
        self.mapbase = Pose().position
        self.cmd = Twist()
        self.map_origin = Pose().position
        self.map_res = 0.0
        self.odata = np.array([])

        # occupancy grid and positions
        self.occ_subscription = self.create_subscription(
            OccupancyGrid,
            'map',
            self.occ_callback,
            qos_profile_sensor_data)
        self.occ_subscription  # prevent unused variable warning
        self.occdata = np.array([])
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer, self)

        # lidar scan
        self.scan_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            qos_profile_sensor_data)
        self.scan_subscription  # prevent unused variable warning
        self.laser_range = np.array([])

        self.tf_subscriber = self.create_subscription(
            Pose, 
            'map2base',
            self.tf_callback, 
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
        self.tf_subscriber
        
    def tf_callback(self, msg):
        rclpy.spin_once(self)
        orientation_quat = msg.orientation
        #quaternion = [orientation_quat.x, orientation_quat.y, orientation_quat.z, orientation_quat.w]
        (self.roll, self.pitch, self.yaw) = euler_from_quaternion(orientation_quat.x, orientation_quat.y, orientation_quat.z, orientation_quat.w)
        self.mapbase = msg.position
        self.pos_x = msg.position.x
        self.pos_y = msg.position.y

    def scan_callback(self, msg):
        rclpy.spin_once(self)
        # self.get_logger().info('In scan_callback')
        # create numpy array
        self.laser_range = np.array(msg.ranges)
        # print to file
        # np.savetxt(scanfile, self.laser_range)
        # replace 0's with nan
        self.laser_range[self.laser_range==0] = np.nan

    def occ_callback(self, msg):
        # create numpy array occdata, WHICH IS 1 DIMENSIONAL -- 
        # the following code makes it into 2D for coordinate systems to make sense
        occdata = np.array(msg.data)

        # New threshold values for the desired conditions
        threshold_value_low = 60
        threshold_value_high = 95  # Set to the same value for exclusive handling

        # Apply thresholding with the updated conditions
        thresholded_data = np.where(occdata == -1,
                                    occdata,  # Leave -1 values unchanged
                                    np.where(occdata == 0, 0,  # Values equal to 0 remain 0
                                            np.where((occdata > 0) & (occdata < threshold_value_low), -1,  # Values between 0 and 60 become -1
                                                    np.where(occdata > threshold_value_high, 100, occdata))))  # Values above 60 become 100, others remain unchanged

        occdata = thresholded_data
    
        # compute histogram to identify bins with -1, values between 0 and below 50, 
        # and values between 50 and 100. The binned_statistic function will also
        # return the bin numbers so we can use that easily to create the image 
        occ_counts, edges, binnum = scipy.stats.binned_statistic(occdata, np.nan, statistic='count', bins=occ_bins)
        self.map_visit = occ_counts[1]
        # get width and height of map
        iwidth = msg.info.width
        iheight = msg.info.height
        self.map_width = iwidth
        self.map_height = iheight

        # calculate total number of bins
        total_bins = iwidth * iheight
        # log the info
        # self.get_logger().info('Unmapped: %i Unoccupied: %i Occupied: %i Total: %i' % (occ_counts[0], occ_counts[1], occ_counts[2], total_bins))
        try:
            trans = self.tfBuffer.lookup_transform('map', 'base_link', rclpy.time.Time())

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().info('No transformation found')
            return
            
        cur_pos = trans.transform.translation
        cur_rot = trans.transform.rotation
        self.pos_x = cur_pos.x
        self.pos_y = cur_pos.y
        # self.get_logger().info('Trans: %f, %f' % (self.pos_x, self.pos_y))
        
        # convert quaternion to Euler angles
        roll, pitch, yaw = euler_from_quaternion(cur_rot.x, cur_rot.y, cur_rot.z, cur_rot.w)
        # self.get_logger().info('Rot-Yaw: R: %f D: %f' % (yaw, np.degrees(yaw)))

        # get map resolution
        map_res = msg.info.resolution
        self.map_res = map_res
        # get map origin struct has fields of x, y, and z
        map_origin = msg.info.origin.position
        self.map_origin = map_origin 
        # get map grid positions for x, y position
        grid_x = round((cur_pos.x - map_origin.x) / map_res)
        grid_y = round(((cur_pos.y - map_origin.y) / map_res))
        self.grid_x = grid_x
        self.grid_y = grid_y
        # self.get_logger().info('Grid Y: %i Grid X: %i' % (grid_y, grid_x))
        # binnum go from 1 to 3 so we can use uint8
        # convert into 2D array using column order
        odata = np.uint8(binnum.reshape(msg.info.height,msg.info.width))
        self.odata = odata

    def stopbot(self):
        self.get_logger().info('stopping in stopbot')
        self.cmd.linear.x = 0.0
        self.cmd.angular.z = 0.0
        self.publisher_.publish(self.cmd)
    
    def movebot(self):
        self.get_logger().info('moving in movebot')
        self.cmd.linear.x = speedchange
        self.cmd.angular.z = 0.0
        self.publisher_.publish(self.cmd)
        
    def rotatebot(self, rot_angle):
        # self.get_logger().info('In rotatebot')
        # create Twist object
        twist = Twist()
        rclpy.spin_once(self)
        # get current yaw angle
        current_yaw = self.yaw
        # log the info
        self.get_logger().info('Current: %f' % math.degrees(current_yaw))
        # we are going to use complex numbers to avoid problems when the angles go from
        # 360 to 0, or from -180 to 180
        c_yaw = complex(math.cos(current_yaw),math.sin(current_yaw))
        # calculate desired yaw
        target_yaw = current_yaw + math.radians(rot_angle)
        # convert to complex notation
        c_target_yaw = complex(math.cos(target_yaw),math.sin(target_yaw))
        self.get_logger().info('Desired: %f' % math.degrees(cmath.phase(c_target_yaw)))
        # divide the two complex numbers to get the change in direction
        c_change = c_target_yaw / c_yaw
        # get the sign of the imaginary component to figure out which way we have to turn
        c_change_dir = np.sign(c_change.imag)
        # set linear speed to zero so the TurtleBot rotates on the spot
        twist.linear.x = 0.0
        # set the direction to rotate
        twist.angular.z = c_change_dir * rotatechange
        # start rotation
        self.publisher_.publish(twist)

        # we will use the c_dir_diff variable to see if we can stop rotating
        c_dir_diff = c_change_dir
        # self.get_logger().info('c_change_dir: %f c_dir_diff: %f' % (c_change_dir, c_dir_diff))
        # if the rotation direction was 1.0, then we will want to stop when the c_dir_diff
        # becomes -1.0, and vice versa
        while(c_change_dir * c_dir_diff > 0):
            # allow the callback functions to run
            rclpy.spin_once(self)
            current_yaw = self.yaw
            # convert the current yaw to complex form
            c_yaw = complex(math.cos(current_yaw),math.sin(current_yaw))
            # self.get_logger().info('Current Yaw: %f' % math.degrees(current_yaw))
            # get difference in angle between current and target
            c_change = c_target_yaw / c_yaw
            # get the sign to see if we can stop
            c_dir_diff = np.sign(c_change.imag)
            # self.get_logger().info('c_change_dir: %f c_dir_diff: %f' % (c_change_dir, c_dir_diff))

        self.get_logger().info('End Yaw: %f' % math.degrees(current_yaw))
        # set the rotation speed to 0
        twist.angular.z = 0.0
        # stop the rotation
        self.publisher_.publish(twist)

    # grid functions
    def check_surroundings(self, array, x, y, radius=1):
        try:
            """
            Check the cells around a given center in a 2D grid.

            Parameters:
            - grid: 2D array representing the grid
            - x, y: Coordinates of the center
            - radius: Radius around the center to check (default is 1)

            Returns:
            - List of values in the surrounding cells
            """
            surroundings = []
            rows = len(self.odata)
            cols = len(self.odata[0])

            for i in range(x - radius, x + radius + 1):
                for j in range(y - radius, y + radius + 1):
                    # Check if the current coordinates are within the grid boundaries
                    if 0 <= i < rows and 0 <= j < cols:
                        surroundings.append(array[i,j]) #i,j flipped?
                    else:
                        # If outside the grid, append a default value (e.g., -1)
                        # surroundings.append(-1)  # or any default value you prefer
                        continue
            return surroundings

        except Exception as e:
            print('in checking surroundings')
            print(e)

    def get_grid(self, x, y):
        rclpy.spin_once(self)
        print('getting grid')
        grid_x = round((x - self.map_origin.x) / self.map_res)
        grid_y = round((y - self.map_origin.y) / self.map_res)
        print(f'gridx, gridy of goal: {grid_x}, {grid_y}')
        return int(grid_x),int(grid_y)   # may have to change back to `return (grid_x, grid_y)`

    def get_pose(self, grid_x, grid_y):
        rclpy.spin_once(self)
        print('getting pose')
        pose_x = grid_x*self.map_res + self.map_origin.x
        pose_y = grid_y*self.map_res + self.map_origin.y
        return pose_x, pose_y

    def search_radius(self, array, x, y, max_radius):
        try:
            # rclpy.spin_once(self)
            frontier = True
            for radius in range(1, max_radius + 1):
                
                # Define the boundaries of the search area for the current radius
                min_x = max(0, x - radius)
                max_x = min(array.shape[1] - 1, x + radius)#shape 1 get columns
                min_y = max(0, y - radius)
                max_y = min(array.shape[0] - 1, y + radius)#shape 0 get rows
                
                # Iterate over the search area and check for the number 2
                # since odata is column ordered, in the for loop 
                # we need to search the columns first...
                for i in range(min_y, max_y + 1):
                    for j in range(min_x, max_x + 1):
                        if array[i, j] == 1:
                            surrounding_cells = self.check_surroundings(array, i, j, 1)
                            # print(surrounding_cells)
                            # check for explored cells which is 2 in odata:
                            
                            if (j,i) in visited:#j is x, i is y
                                #print('visited',(j,i))
                                continue
                            
                            if 2 in surrounding_cells and 3 not in surrounding_cells and surrounding_cells.count(1)>2:
                                return True, (j,i)  # If found, return True and the tuple of coordinates
            return False, (max_radius,'no frontier') # If not found within max_radius, return False and max_radius

        except Exception as e:
            print('in search radius')
            print(e)

    def spin_test(self):
        time.sleep(0.6)
        rclpy.spin_once(self)
        time.sleep(0.6)
        rclpy.spin_once(self)

        timeout = time.time() + 5
        while True:
            rclpy.spin_once(self)
            if time.time()>timeout:
                break

    def search_frontiers(self):
        rclpy.spin_once(self)
        try: 
            print('searching for frontiers...')
            rclpy.spin_once(self)
            time.sleep(0.2)
            grid = self.odata
            # print(grid)
            #change robot coordinates to the grid
            #robot_grid = self.get_grid(self.pos_x, self.pos_y)
            #print(f'robot\'s grid coordinates: {robot_grid}')

            # look for frontiers around the robot grid
            #start search at robot grid position 
            found, coor = self.search_radius(grid, self.grid_x, self.grid_y, 150 )
            print(found, coor)
            goal_x, goal_y = self.get_pose(coor[0], coor[1])
            print(f'goal x and y in RW Pose: {goal_x}, {goal_y}')
            return goal_x, goal_y, found

        except Exception as e:
            print('tried searching frontiers but got Exception:')
            print(e)

    def pick_direction(self):
        rclpy.spin_once(self)
        self.get_logger().info('In pick_direction')
        deadend = False
        angles = []
        angle = 0
        print(self.laser_range.size)
        if self.laser_range.size != 0:
            length_laser_array = len(self.laser_range) #which will be less than 360 ~250+
            # print(f'laser range arraY: {self.laser_range}')
            # print(f'length of array: {length_laser_array}')
            
            search_distances = []

            for i in search_angles:
                #do some transformation to get estimated angles and index in the laser_range
                index = int( i*length_laser_array/360)
                search_distances.append(self.laser_range[index])

            pairs = dict(zip(search_angles, search_distances))
            # print(f'pairs of angle/distance: {pairs}')
            
            for a in pairs:
                if pairs[a]>search_distance and a not in angles:
                    angles.append(abs(a))

            print(f'angles available: {angles}')
            rclpy.spin_once(self)

            #if there are angles within the 
            if len(angles)!=0:
                angle = min(angles, key=abs)   #to avoid error
                for a in sorted(angles):
                    if pairs[a] == pairs[-a]:
                        continue
                    elif pairs[-a] > pairs[a]:
                        angle = -a -30
                        # break out of the loop if you want to search from the centre angle
                        # then remember to add offset
                        break
                    elif pairs[a] > pairs[-a]:
                        angle = a +30#+ some offset
                        break
                        
                #angle= min(abs(angle) for angle in angles if angle!=0)
                print(f'angle: {angle}')
                time.sleep(0.3)

            #
            else:
                rclpy.spin_once(self)
                
                #check to the left of the turtlebot, and see if the distance is more  to the left than to the right
                rightAngle = int(90*length_laser_array/360)
                total = 0
                counter = 0
                for i in range(rightAngle-2,rightAngle+3):
                    if np.isnan(self.laser_range[i]):
                        continue
                    else:
                        total += self.laser_range[i]
                        counter += 1
                if counter != 0:
                    left_index = total/counter
                else:
                    left_index = 0 #check for zero error

                total = 0
                counter = 0
                for i in range(-rightAngle-2,-rightAngle+3):
                    if np.isnan(self.laser_range[i]):
                        continue
                    else:
                        total += self.laser_range[i]
                        counter += 1
                if counter != 0:
                    right_index = total/counter
                else:
                    right_index = 0 #check for zero error

                print(f'left {left_index}, right {right_index}')
                if left_index > search_distance:
                    if left_index > right_index:
                        angle = 90
                        print('LEFT!!!')
                        print(f'distance to the left: {left_index}')
                        print(f'distance to the right: {right_index}')
                        rclpy.spin_once(self)
                elif right_index > search_distance:
                    angle = -90
                    print('RIGHT@@!!')
                    print(f'distance to the left: {left_index}')
                    print(f'distance to the right: {right_index}')
                    rclpy.spin_once(self)
                else:
                    print('reverse')
                    lidar = np.nanargmax(self.laser_range) #reverse to original path
                    angle = lidar/len(self.laser_range)*360
                    if angle>0:
                        angle += 10
                    elif angle>180:
                        angle -= 10
                    deadend = True
                    print(angle)

        else:
            #reverse
            angle = 180
            self.get_logger().info('No data!')

        rclpy.spin_once(self)
        
        # rotate to that direction
        self.rotatebot(float(angle))
        time.sleep(0.3)
        rclpy.spin_once(self)
    
        # start moving
        self.get_logger().info('avoid')
        
        #CHECK THIS
        rclpy.spin_once(self)
        lri = (self.laser_range[front_angles]<float(stop_distance)).nonzero()
        if len(lri[0])==0:
            print('direction clear!')
            self.movebot()
        else:
            self.pick_direction()

        while len(lri[0])==0:
            print(f'avoiding pos,yaw:{self.pos_x},{self.pos_y},{self.yaw}')
            lri = (self.laser_range[front_angles]<float(stop_distance)).nonzero()
            rclpy.spin_once(self)
            
            if ((self.pos_y < door_y + box_thres and self.pos_y > door_y - box_thres) 
               and (self.pos_x < door_x + box_thres and self.pos_x > door_x - box_thres)):
                print('robot at door')
                self.stopbot()
                break

            if (len(lri[0])>0):
                self.stopbot()
                print('stopped avoiding in pick direction')
                if deadend:
                    print('getting out deadend')
                    self.pick_direction()
                    deadend = False
                break
    

    def rotateTo(self, x_goal, y_goal):
        try: 
            rclpy.spin_once(self)
            x1, y1, x2, y2, current_yaw = self.pos_x, self.pos_y, x_goal, y_goal, self.yaw
            print(f'rotating to goal: {x_goal}, {y_goal}')
            print(f'current robot coordinates and yaw: {x1}, {y1}, {self.yaw}')
            yaw_difference, distance = calculate_yaw_and_distance(x1, y1, x2, y2, current_yaw)
            yaw_difference = yaw_difference / math.pi * 180
            self.rotatebot(yaw_difference)
            rclpy.spin_once(self)

        except Exception as e:
            print('in rotateTo')
            print(e)
            

    def moveTo(self,x_goal,y_goal):
        try:
            rclpy.spin_once(self)
            x1, y1, x2, y2 = self.pos_x, self.pos_y, x_goal, y_goal
            print(f'current robot coordinates and yaw: {x1}, {y1}, {self.yaw}')
            counter = 0
            x_goal_grid,y_goal_grid = self.get_grid(x_goal,y_goal)
            goalx,goaly = x_goal,y_goal
            while not ((self.pos_y < y2 + box_thres and self.pos_y > y2 - box_thres) 
               and (self.pos_x < x2 + box_thres and self.pos_x > x2 - box_thres)):
                rclpy.spin_once(self)                         
                
                rad,dis = calculate_yaw_and_distance(self.pos_x, self.pos_y, x_goal, y_goal, self.yaw)
                angle  = int(math.degrees(rad))
                target_angle = int((angle-self.yaw)/360*len(self.laser_range))
                print(f'distance to target, lidar distance to target {dis},{self.laser_range[target_angle]}')
                
                #remove this CHUNK
                lri = (self.laser_range[range(target_angle-5,target_angle+6)]<float(dis+free_dis_thres)).nonzero()
                if not np.isnan(self.laser_range[target_angle]):
                    if(len(lri[0])==0):
                        print('target is free')
                        self.rotatebot(angle-self.yaw)
                elif dis>1.5:
                    print('chcecking target faraway')
                    total = 0
                    num = 0
                    for i in range(target_angle-5,target_angle+6):
                        if np.isnan(self.laser_range[i]):
                            continue
                        else:
                            total += self.laser_range[i]
                            num += 1
                    if num != 0:
                        target_index = total/num
                    else:
                        target_index = 0
                    if target_index>1:
                        print('free distance to target faraway')
                        self.rotatebot(angle-self.yaw)
                
                rclpy.spin_once(self)
                
                print(f'trying to move to goal... {x2}, {y2}')
        
                # before moving if there is stuff in front, avoid it
                lri = (self.laser_range[front_angles]<float(stop_distance)).nonzero()
                if len(lri[0])!=0:
                    print('but target is blocked')
                    self.pick_direction()
                else:
                    self.movebot()

                lri = (self.laser_range[front_angles]<float(stop_distance)).nonzero()
                while len(lri[0])==0:
                    print(f'current robot coordinates and yaw: {self.pos_x}, {self.pos_y}, {self.yaw}')
                    if goalx == door_x and goaly == door_y:
                        if ((self.pos_y < door_y + box_thres and self.pos_y > door_y - box_thres) 
                    and (self.pos_x < door_x + box_thres and self.pos_x > door_x - box_thres)):
                            print('robot at door')
                            self.stopbot()
                            break
                    rclpy.spin_once(self)
                    lri = (self.laser_range[front_angles]<float(stop_distance)).nonzero()
                    if(len(lri[0])>0):
                        self.stopbot()
                        break
                
                # if the list is not empty
                counter += 1
                rclpy.spin_once(self)

                #COMMENT THIS OUT I THINK UNNECESSARY
                self.rotateTo(goalx, goaly)
    
                 #CHANGE THIS TO DETERMINE HOW MANY TIMES TO AVOID   
                if counter > 1:
                    break
                
        except Exception as e:
            print('exception in moveTo function: ')
            print(e)
        finally:
        	# stop moving
            self.stopbot()

    def exploration(self):
        try:
            map_bins = height_map/self.map_res * width_map/self.map_res
            while rclpy.ok():
                rclpy.spin_once(self)
                print('visit %', self.map_visit/map_bins)
                if self.map_visit/map_bins < map_percent:
                    goalx, goaly, found = self.search_frontiers()
                    if found == False:
                        goalx, goaly = door_x, door_y

                #going to door
                else:
                    print('moving to door')
                    goalx, goaly = door_x, door_y

                    # door reached... 
                    if ((self.pos_y < door_y + box_thres and self.pos_y > door_y - box_thres) 
                       and (self.pos_x < door_x + box_thres and self.pos_x > door_x - box_thres)):
                        
                        '''calibration, check x and y and make sure it is accurate'''
                        
                        rclpy.spin_once(self)
                        print(f'current yaw: {math.degrees(self.yaw)}')
                        print('ADJUSTING ANGLE AT DOOR')
                        self.rotatebot(90-math.degrees(self.yaw))
                        time.sleep(1)
                        
                        # URL to which the request will be sent
                        url = (f"http://{esp32_ip}/openDoor")
                        
                        # Data to send in the request (if any)
                        data = {"action": "openDoor", "parameters": {"robotId": "{TurtleBot3_ID}"}}
                        
                        # Send HTTP POST request
                        while True:
                            response = requests.post(url, json=data)
                            time.sleep(1)
                            # Print the response received from ESP32 server
                            print("Response from ESP32:", response.text) 
                            door_ans = response.json()["data"]["message"]
                            if door_ans == "door1":
                                self.rotatebot(90)
                                break
                            elif door_ans == "door2":
                                self.rotatebot(-90)
                                break
                            else:
                                time.sleep(60)
                                response = requests.post(url, json=data)

                        time.sleep(4)
                        twist = Twist()
                        twist.linear.x = speedchange
                        twist.angular.z = 0.0
                        self.publisher_.publish(twist)
                        
                        time.sleep(3)
        
                        
                        timeout = time.time()+door_time
                        while time.time()<timeout:  
                            rclpy.spin_once(self)
                            lri = (self.laser_range[range(-10,10)]<float(stop_distance)).nonzero()
                            if(len(lri[0])!=0):
                                print('bucket in front')
                                self.stopbot()

                        self.stopbot()

                        #find bucket
                        rclpy.spin_once(self)
                        time.sleep(1)
                        print(self.laser_range)
                        if self.laser_range.size != 0:
                            dis = 100
                            for i in range(-70,70):
                                ang = int(i/360*len(self.laser_range))
                                if dis>self.laser_range[ang]:
                                    dis = self.laser_range[ang]
                                    lr2i = i
                        else:
                            lr2i = 0
                            self.get_logger().info('No data!')

                        # rotate to that direction
                        print(lr2i)
                        self.rotatebot(float(lr2i)-self.yaw)

                        # start moving
                        self.get_logger().info('Start moving')
                        twist = Twist()
                        twist.linear.x = 0.05
                        twist.angular.z = 0.0
                        # not sure if this is really necessary, but things seem to work more
                        # reliably with this
                        
                        self.publisher_.publish(twist)
                        rclpy.spin_once(self)
                        
                        while self.laser_range[0] > dist_threshold:
                            rclpy.spin_once(self)
                        
                        self.stopbot()

                        #launching!
                        launch_ball()
                        break
                        
                print(f'robot coordinates, yaw: {self.pos_x}, {self.pos_y}, {self.yaw}')
                print(f'goalx, goaly: {goalx}, {goaly}')
                
                self.rotateTo(goalx, goaly)
                rclpy.spin_once(self)
                
                lri = (self.laser_range[front_angles]<float(stop_distance)).nonzero()
                # if the list is not empty
                if(len(lri[0])>0):
                    print('obstacles detected')
                    rclpy.spin_once(self)
                    self.pick_direction()
                
                else:
                    self.moveTo(goalx, goaly)

                x,y = self.get_grid(goalx, goaly)
                
                for i in range(x-2,x+3):
                    for j in range(y-2,y+3):
                        visited.add((i,j))
                print('added',x,y)
                # print('vnumber of visited points',len(visited))
                rclpy.spin_once(self)
                
        except Exception as e:
            print('in mover')
            print(e)
        # Ctrl-c detected
        finally:
            # stop moving
            self.stopbot()
            self.get_logger().info('end robot coordinates -- x,y: %f, %f' % (self.mapbase.x, self.mapbase.y))
            print('mission finished! SU!')

def main(args=None):
    rclpy.init(args=args)
    solver = Solver() 
    solver.spin_test()
    solver.movebot()
    time.sleep(startMove)
    solver.stopbot()    
    
    solver.exploration()
    GPIO.cleanup()
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    solver.destroy_node()
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()
