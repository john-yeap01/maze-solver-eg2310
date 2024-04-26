"""
Created on Tue Mar 12 12:50:49 2024

@author: johnyeap01
"""
# combines the moverotate and occ2 scripts from Dr Yen's repo


import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import ReliabilityPolicy, QoSProfile
from rclpy.qos import qos_profile_sensor_data
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import matplotlib.pyplot as plt
from PIL import Image
import math
import cmath
import numpy as np
import scipy.stats
import time

# constants
box_thres = 0.1
rotate_change = 0.35
slow_rotate_change = 0.10
speed_change= 0.10
rotatechange = 0.1
speedchange = 0.1
time_threshold = 1.28 * box_thres / speed_change
dist_threshold = 0.18        # Distance threshold for the robot to stop in front of the pail
initial_yaw = 0.0
front_angle = 3
front_angle_6 = 65
front_angle_range = range(-front_angle,front_angle+1,1)
stop_distance = 0.25
occ_bins = [-1, 0, 50, 100]
map_bg_color = 1
ROTATE_CHANGE =  0.4
ANGLE_THRESHOLD = 0.8
map_res = 0.05


# code from https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
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

# function to check if keyboard input is a number as
# isnumeric does not handle negative numbers
def isnumber(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


# class for moving and rotating robot
class Mover(Node):
    def __init__(self):
        super().__init__('movethere')
        self.publisher_ = self.create_publisher(
            Twist,
            'cmd_vel',
            10)
        
        # self.get_logger().info('Created publisher')
        #self.odom_subscription = self.create_subscription(
        #    Odometry,
        #   'odom',
        #    self.odom_callback,
        #    10)
        # self.get_logger().info('Created subscriber')
        # self.odom_subscription  # prevent unused variable warning
        
        self.occ_subscription = self.create_subscription(
            OccupancyGrid,
            'map',
            self.occupancy_callback,
            qos_profile_sensor_data)
        self.occ_subscription  # prevent unused variable warning
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer, self)
        
        
        self.map_frame_subscriber = self.create_subscription(
                Pose, 
                'map2base',
                self.map_callback, 
                QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
        self.map_frame_subscriber


        self.pos_x = 0.0
        self.pos_y = 0.0
        self.yaw = 0.0
        self.mapbase = Pose().position
        self.cmd = Twist()
        self.map_origin = []
        self.map_res = 0.05
        
        # initialize variables
        #self.roll = 0
        #self.pitch = 0
        #self.yaw = 0


    # function to set the class variables using the odometry information
    #def odom_callback(self, msg):
        # self.get_logger().info(msg)
        # self.get_logger().info('In odom_callback')
    #    orientation_quat =  msg.pose.pose.orientation
    #    self.roll, self.pitch, self.yaw = euler_from_quaternion(orientation_quat.x, orientation_quat.y, orientation_quat.z, orientation_quat.w)

    
    def map_callback(self, msg):
        rclpy.spin_once(self)
        orientation_quat = msg.orientation
        #quaternion = [orientation_quat.x, orientation_quat.y, orientation_quat.z, orientation_quat.w]
        (self.roll, self.pitch, self.yaw) = euler_from_quaternion(orientation_quat.x, orientation_quat.y, orientation_quat.z, orientation_quat.w)
        self.mapbase = msg.position
        self.pos_x = msg.position.x
        self.pos_y = msg.position.y
        
    def occupancy_callback(self, msg):
        # create numpy array
        occdata = np.array(msg.data)
        # compute histogram to identify bins with -1, values between 0 and below 50, 
        # and values between 50 and 100. The binned_statistic function will also
        # return the bin numbers so we can use that easily to create the image 
        occ_counts, edges, binnum = scipy.stats.binned_statistic(occdata, np.nan, statistic='count', bins=occ_bins)
        # get width and height of map
        iwidth = msg.info.width
        iheight = msg.info.height
        # calculate total number of bins
        total_bins = iwidth * iheight
        # log the info
        # self.get_logger().info('Unmapped: %i Unoccupied: %i Occupied: %i Total: %i' % (occ_counts[0], occ_counts[1], occ_counts[2], total_bins))

        # find transform to obtain base_link coordinates in the map frame
        # lookup_transform(target_frame, source_frame, time)
        try:
            trans = self.tfBuffer.lookup_transform('map', 'base_link', rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().info('No transformation found')
            return
            
        cur_pos = trans.transform.translation
        cur_rot = trans.transform.rotation
        self.get_logger().info('Trans: %f, %f' % (cur_pos.x, cur_pos.y))
        
        # convert quaternion to Euler angles
        roll, pitch, yaw = euler_from_quaternion(cur_rot.x, cur_rot.y, cur_rot.z, cur_rot.w)
        # self.get_logger().info('Rot-Yaw: R: %f D: %f' % (yaw, np.degrees(yaw)))

        # get map resolution
        map_res = msg.info.resolution
        self.map_res = map_res
        # get map origin struct has fields of x, y, and z
        map_origin = msg.info.origin.position
        self.map_origin  = map_origin
        # get map grid positions for x, y position
        grid_x = round((cur_pos.x - map_origin.x) / map_res)
        grid_y = round(((cur_pos.y - map_origin.y) / map_res))
        self.get_logger().info('Grid Y: %i Grid X: %i' % (grid_y, grid_x))

        # binnum go from 1 to 3 so we can use uint8
        # convert into 2D array using column order
        odata = np.uint8(binnum.reshape(msg.info.height,msg.info.width))
        # set current robot location to 0
        odata[grid_y][grid_x] = 0
      
    def stopbot(self, delay):
        self.cmd.linear.x = 0.0
        self.cmd.angular.z = 0.0
        self.publisher_.publish(self.cmd)


    
    #higher accuracy than rotatebot apparently
    def rotate_to(self, degree):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher_.publish(twist)
        degree = float(degree)
        desired_position = math.degrees(self.yaw) + degree
        if (desired_position > 180):
            desired_position -= 360
        elif (desired_position < -180):
            desired_position += 360

        # self.get_logger().info('Current angle diff: %f' % curr_angle_diff)

        if (degree > 0):
            twist.angular.z += ROTATE_CHANGE
        else:
            twist.angular.z -= ROTATE_CHANGE
        self.publisher_.publish(twist)

        self.get_logger().info('degree: %s' % str(degree))
        while (abs(desired_position - math.degrees(self.yaw)) > ANGLE_THRESHOLD):
            if(abs(desired_position - math.degrees(self.yaw)) <= 10 and twist.angular.z == ROTATE_CHANGE):
                twist.angular.z = 0.25 * twist.angular.z
                self.publisher_.publish(twist)
            self.get_logger().info('desired - yaw: %s' % str(abs(desired_position - math.degrees(self.yaw))))
            rclpy.spin_once(self)

    # function to rotate the TurtleBot
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
        twist.angular.z = c_change_dir * speedchange
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
            self.get_logger().info('Current Yaw: %f' % math.degrees(current_yaw))
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
    
    # another Move program:
    def MoveForward(self, x_coord, y_coord):
        
        self.get_logger().info('Moving Forward...')
        #makes sure the robot lands within a boxed range
        while not ((self.mapbase.y < y_coord + box_thres and self.mapbase.y > y_coord - box_thres) 
                   and (self.mapbase.x < x_coord + box_thres and self.mapbase.x > x_coord - box_thres)):
            rclpy.spin_once(self)
            # self.get_logger().info('I receive "%s"' % str(self.mapbase.y))
            self.cmd.linear.x = speed_change
            self.cmd.angular.z = 0.0
            self.publisher_.publish(self.cmd)

        self.Clocking()
        self.cmd.linear.x = 0.0
        self.publisher_.publish(self.cmd)

        
    def moveTo(self):
        twist = Twist()
        
        try:
            while True:
                x_goal = float(input("X coordinate of goal: "))
                y_goal = float(input("Y coordinate of goal: "))
                rclpy.spin_once(self)
                x1, y1, x2, y2, current_yaw = self.mapbase.x, self.mapbase.y, x_goal, y_goal, self.yaw
                yaw_difference, distance = calculate_yaw_and_distance(x1, y1, x2, y2, current_yaw)
                
                yaw_difference = yaw_difference / math.pi * 180
                
                duration = distance/speedchange
                #self.rotatebot(yaw_difference)
                self.rotatebot(yaw_difference)
                self.stopbot(0.1)
                while not ((self.mapbase.y < y2 + box_thres and self.mapbase.y > y2 - box_thres) 
                   and (self.mapbase.x < x2 + box_thres and self.mapbase.x > x2 - box_thres)):
                    rclpy.spin_once(self)
                    # self.get_logger().info('I receive "%s"' % str(self.mapbase.y))
                    self.cmd.linear.x = speed_change
                    self.cmd.angular.z = 0.0
                    self.publisher_.publish(self.cmd)
            
                
                self.stopbot(0.5)
                rclpy.spin_once(self)
                self.get_logger().info('end robot coordinates -- x,y: %f, %f' % (self.mapbase.x, self.mapbase.y))
                self.get_logger().info('end robot yaw: %f' % (math.degrees(self.yaw)))
                
        
        except Exception as e:
            print('exception in moveTo function: ')
            print(e)
            
        finally:
        	# stop moving
            self.stopbot(3)
                

    
# function to read keyboard input
    def readKey(self):
        twist = Twist()
        try:
            while True:
                # get keyboard input
                cmd_char = str(input("Keys w/x/a/d -/+int s: "))
                
                # use our own function isnumber as isnumeric 
                # does not handle negative numbers
                if isnumber(cmd_char):
                    # rotate by specified angle
                    self.rotatebot(int(cmd_char))
                else:
                    # check which key was entered
                    if cmd_char == 's':
                        # stop moving
                        twist.linear.x = 0.0
                        twist.angular.z = 0.0
                    elif cmd_char == 'w':
                        # move forward
                        twist.linear.x += speedchange
                        twist.angular.z = 0.0
                    elif cmd_char == 'x':
                        # move backward
                        twist.linear.x -= speedchange
                        twist.angular.z = 0.0
                    elif cmd_char == 'a':
                        # turn counter-clockwise
                        twist.linear.x = 0.0
                        twist.angular.z += rotatechange
                    elif cmd_char == 'd':
                        # turn clockwise
                        twist.linear.x = 0.0
                        twist.angular.z -= rotatechange
                        
                    # start the movement
                    self.publisher_.publish(twist)
    
        except Exception as e:
            print('exception in readkey func')
            print(e)
            
		# Ctrl-c detected
        finally:
        	# stop moving
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.publisher_.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    
    mover = Mover()    
    
    mover.moveTo()
    

    # rclpy.spin(mover)
    
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    mover.destroy_node()
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()