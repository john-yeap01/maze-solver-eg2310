import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
import numpy as np
import math 


class Scanner(Node):

    def __init__(self):
        super().__init__('scanner')
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.listener_callback,
            qos_profile_sensor_data)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        # create numpy array
        laser_range = np.array(msg.ranges)
        # replace 0's with nan
        laser_range[laser_range==0] = np.nan
        # find index with minimum value
        lr2i = np.nanargmin(laser_range)
        length_laser_array = len(laser_range)
        print(f'lr2i:{lr2i}')
        print(f'laser range arraY: {laser_range}')
        print(f'length of array: {length_laser_array}')
        print(f'Shortest distance at {lr2i*360/length_laser_array} degrees')
        # numpy.nanmin()function is used when to returns minimum value of an array 
        # or along any specific mentioned axis of the array, ignoring any Nan value.
        # nanargmin>

        # log the info
        #self.get_logger().info('Shortest distance at %i degrees' % lr2i)


def main(args=None):
    rclpy.init(args=args)

    scanner = Scanner()

    rclpy.spin(scanner)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    scanner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
