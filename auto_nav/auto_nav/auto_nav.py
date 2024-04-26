import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
import numpy as np
import math
import cmath
import time

# constants
rotatechange = 0.2
speedchange = 0.1
occ_bins = [-1, 0, 100, 101] #??
stop_distance = 0.25
search_distance = 0.5
front_angle = 10
front_angles = range(-front_angle,front_angle+1,1)
search_angle = 60
search_angles = range(-search_angle,search_angle+1,5)
scanfile = 'lidar.txt'
mapfile = 'map.txt'
#offset

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

class AutoNav(Node):

    def __init__(self):
        super().__init__('auto_nav')
        
        # create publisher for moving TurtleBot
        self.publisher_ = self.create_publisher(Twist,'cmd_vel',10)
        # self.get_logger().info('Created publisher')
        
        # create subscription to track orientation
        self.odom_subscription = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10)
        # self.get_logger().info('Created subscriber')
        self.odom_subscription  # prevent unused variable warning
        # initialize variables
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        
        # create subscription to track occupancy
        self.occ_subscription = self.create_subscription(
            OccupancyGrid,
            'map',
            self.occ_callback,
            qos_profile_sensor_data)
        self.occ_subscription  # prevent unused variable warning
        self.occdata = np.array([])
        
        # create subscription to track lidar
        self.scan_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            qos_profile_sensor_data)
        self.scan_subscription  # prevent unused variable warning
        self.laser_range = np.array([])


    def odom_callback(self, msg):
        # self.get_logger().info('In odom_callback')
        orientation_quat =  msg.pose.pose.orientation
        self.roll, self.pitch, self.yaw = euler_from_quaternion(orientation_quat.x, orientation_quat.y, orientation_quat.z, orientation_quat.w)


    def occ_callback(self, msg):
        # self.get_logger().info('In occ_callback')
        # create numpy array
        occdata = np.array(msg.data)
        # compute histogram to identify percent of bins with -1
        occ_counts = np.histogram(occdata,occ_bins)
        # calculate total number of bins
        total_bins = msg.info.width * msg.info.height
        # log the info
        # self.get_logger().info('Unmapped: %i Unoccupied: %i Occupied: %i Total: %i' % (occ_counts[0][0], occ_counts[0][1], occ_counts[0][2], total_bins))

        # make occdata go from 0 instead of -1, reshape into 2D
        oc2 = occdata + 1
        # reshape to 2D array using column order
        # self.occdata = np.uint8(oc2.reshape(msg.info.height,msg.info.width,order='F'))
        self.occdata = np.uint8(oc2.reshape(msg.info.height,msg.info.width))
        # for row in self.occdata:

        #     print(row)
        # print to file
        # np.savetxt(mapfile, self.occdata)


    def scan_callback(self, msg):
        # self.get_logger().info('In scan_callback')
        # create numpy array
        self.laser_range = np.array(msg.ranges)
        # print to file
        # np.savetxt(scanfile, self.laser_range)
        # replace 0's with nan
        self.laser_range[self.laser_range==0] = np.nan


    # function to rotate the TurtleBot
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


    def pick_direction(self):
        rclpy.spin_once(self)
        
        self.get_logger().info('In pick_direction')
        angles = []
        angle = 0
        if self.laser_range.size != 0:
            length_laser_array = len(self.laser_range)
            print(f'laser range arraY: {self.laser_range}')
            # print(f'length of array: {length_laser_array}')
            try_angles = search_angles
            search_distances = []

            for i in try_angles:
                #do some transformation to get estimated angles and index in the laser_range
                index = int( i*length_laser_array/360)
                search_distances.append(self.laser_range[index])

            pairs = dict(zip(try_angles, search_distances))
            print(f'pairs of angle/distance: {pairs}')
            rclpy.spin_once(self)

            # search through angles for which the distances are more than specific distance
            for a in pairs:
                if pairs[a]>search_distance and a not in angles:
                    angles.append(abs(a))
            if len(angles)!=0:
                #angle = min(angles, key=abs)
                for a in sorted(angles):
                    # if a == 0:
                    #     break
                    if pairs[a] == pairs[-a]:
                        continue
                    elif pairs[-a] > pairs[a]:
                        #print(f'distance for pairs[-a] is {pairs[-a]}')
                        angle = -a - 30
                        break
                    elif pairs[a]>pairs[-a]:
                        angle = a+ 30
                        break
                #angle= min(abs(angle) for angle in angles if angle!=0)
                print(f'angle: {angle}')
                time.sleep(0.3)
            else:
                rclpy.spin_once(self)
                #check to the left of the turtlebot, and see if the distance is more  to the left than to the right
                left_index = int(90*length_laser_array/360)
                right_index = int(-90*length_laser_array/360)

                if self.laser_range[left_index] > self.laser_range[right_index]:
                    angle = 120
                    print('LEFT!!!')
                else: 
                    angle = -120
                    print('RIGHT@@!!')
                    

        else:
            #reverse
            angle = 180
            self.get_logger().info('No data!')

        print('turning!')
        rclpy.spin_once(self)
        print(f'angles available: {angles}')
        # rotate to that direction
        self.rotatebot(float(angle))
        time.sleep(0.5)

        # start moving
        self.get_logger().info('Start moving')
        twist = Twist()
        twist.linear.x = speedchange
        twist.angular.z = 0.0
        # not sure if this is really necessary, but things seem to work more
        # reliably with this
        time.sleep(1)
        self.publisher_.publish(twist)

        


    def stopbot(self):
        self.get_logger().info('In stopbot')
        # publish to cmd_vel to move TurtleBot
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        # time.sleep(1)
        self.publisher_.publish(twist)

        
    def mover(self):
        try:
            # find direction with the largest distance from the Lidar,
            # rotate to that direction, and start moving
            self.pick_direction()
            

            while rclpy.ok():
                rclpy.spin_once(self)
                lri = (self.laser_range[front_angles]<float(stop_distance)).nonzero()
                while len(lri[0])==0:
                    print('AVOIDING')
                    rclpy.spin_once(self)
                    lri = (self.laser_range[front_angles]<float(stop_distance)).nonzero()
                
                    if (len(lri[0])>0):
                        print('STOPPING')
                        self.stopbot()
                        break
                
                # allow the callback functions to run
                rclpy.spin_once(self)

                self.pick_direction()
                
                
            

        except Exception as e:
            print(e)
        
        # Ctrl-c detected
        finally:
            # stop moving
            self.stopbot()

    

def main(args=None):
    rclpy.init(args=args)

    auto_nav = AutoNav()
    #rclpy.spin(auto_nav)
    #key function:
    auto_nav.mover()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    auto_nav.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
