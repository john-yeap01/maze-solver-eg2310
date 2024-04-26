import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import math
import cmath
import time
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from rclpy.qos import ReliabilityPolicy, QoSProfile

#rpi's lidar package, fullrun.py, shortcut is turtle

rotatechange = 0.3
speedchange= 0.15
door_time = 5

dist_threshold = 0.165 # Distance threshold for the robot to stop in front of the pail

esp32_ip = '192.168.38.87'
#esp32_ip = '192.168.227.169' #'192.168.38.169'  # RMB TO CHANGE this to the IP of your ESP32
#esp32_ip = '192.168.38.169'
TurtleBot3_ID = 'turtlebot'

import requests
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)##rpi , lidar, bucket

servo_pin = 13

GPIO.setup(servo_pin, GPIO.OUT)
pwm = GPIO.PWM(servo_pin, 50)
pwm.start(7.5)#90
#pwm.ChangeDutyCycle(2.5)#0
time.sleep(0.5)

def launch_ball():
        print("Launching! Big Balls")
        duty_cycle = 2.5+  float(180) / 18#rotate 90
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

class Solver(Node):
    def __init__(self):
        super().__init__('solver')
        # create publisher for moving TurtleBot
        self.publisher_ = self.create_publisher(Twist,'cmd_vel',10)
        
        # initialize variables
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.pos_x = 0.0
        self.pos_y = 0.0

        self.cmd = Twist()
        self.map_origin = Pose().position

        # create subscription to track lidar
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
        orientation_quat = msg.orientation
        #quaternion = [orientation_quat.x, orientation_quat.y, orientation_quat.z, orientation_quat.w]
        (self.roll, self.pitch, self.yaw) = euler_from_quaternion(orientation_quat.x, orientation_quat.y, orientation_quat.z, orientation_quat.w)
        self.mapbase = msg.position
        self.pos_x = msg.position.x
        self.pos_y = msg.position.y

    def scan_callback(self, msg):
        # self.get_logger().info('In scan_callback')
        # create numpy array
        self.laser_range = np.array(msg.ranges)
        # print to file
        # np.savetxt(scanfile, self.laser_range)
        # replace 0's with nan
        self.laser_range[self.laser_range==0] = np.nan
    
        

    def stopbot(self):
        self.cmd.linear.x = 0.0
        self.cmd.angular.z = 0.0
        self.publisher_.publish(self.cmd)
    
    def movebot(self):
        self.cmd.linear.x = speedchange
        self.cmd.angular.z = 0.0
        self.publisher_.publish(self.cmd)
        
    def rotatebot(self, rot_angle):
        self.get_logger().info('In rotatebot')
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
            print('in loop')
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
    
    def bucket(self):
        try:
            print('moving to door')

            rclpy.spin_once(self)

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
                time.sleep(1)
                door_ans = response.json()["data"]["message"]
                if door_ans == "door1":
                    self.rotatebot(89)
                    break
                elif door_ans == "door2":
                    self.rotatebot(-89)
                    break
                else:
                    time.sleep(60)
                    response = requests.post(url, json=data)
            time.sleep(0.5)
            twist = Twist()
            twist.linear.x = speedchange
            twist.angular.z = 0.0
            # not sure if this is really necessary, but things seem to work more
            # reliably with this
            
            self.publisher_.publish(twist)
            #time.sleep(door_time)# adjus the time needed to move to the centre of the roomS
            time.sleep(3)            
            timeout = time.time()+door_time
            while time.time()<timeout:
                rclpy.spin_once(self)
                lri = (self.laser_range[range(-15,15)]<float(0.3)).nonzero()
                if(len(lri[0])!=0):
                    print('bucket in front')
                    self.stopbot()

            self.stopbot()
            time.sleep(1)
            rclpy.spin_once(self)
            print(self.laser_range)
            if self.laser_range.size != 0:
                dis = 100
                for i in range(-80,81):
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

            launch_ball()
                
        except Exception as e:
            print('in mover')
            print(e)
        # Ctrl-c detected
        finally:
            # stop moving
            self.stopbot()
            print('mission finished!')

def main(args=None):
    rclpy.init(args=args)
    
    solver = Solver() 
    solver.bucket()
    GPIO.cleanup()
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    solver.destroy_node()
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()
