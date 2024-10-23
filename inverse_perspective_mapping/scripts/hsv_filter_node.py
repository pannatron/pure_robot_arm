#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class HSVFilterNode(Node):
    def __init__(self):
        super().__init__('hsv_filter_node')
        self.subscription = self.create_subscription(
            Image,
            '/ipm/image',
            self.image_callback,
            2)
        self.publisher_ = self.create_publisher(Image, '/filtered_image', 10)
        self.br = CvBridge()

        cv2.namedWindow('Trackbars')
        cv2.createTrackbar('H_min', 'Trackbars', 0, 179, self.nothing)
        cv2.createTrackbar('H_max', 'Trackbars', 179, 179, self.nothing)
        cv2.createTrackbar('S_min', 'Trackbars', 0, 255, self.nothing)
        cv2.createTrackbar('S_max', 'Trackbars', 255, 255, self.nothing)
        cv2.createTrackbar('V_min', 'Trackbars', 0, 255, self.nothing)
        cv2.createTrackbar('V_max', 'Trackbars', 255, 255, self.nothing)

        self.get_logger().info('HSV Filter Node has been started.')

    def nothing(self, x):
        pass

    def image_callback(self, msg):
        current_frame = self.br.imgmsg_to_cv2(msg)

        h_min = cv2.getTrackbarPos('H_min', 'Trackbars')
        h_max = cv2.getTrackbarPos('H_max', 'Trackbars')
        s_min = cv2.getTrackbarPos('S_min', 'Trackbars')
        s_max = cv2.getTrackbarPos('S_max', 'Trackbars')
        v_min = cv2.getTrackbarPos('V_min', 'Trackbars')
        v_max = cv2.getTrackbarPos('V_max', 'Trackbars')

        hsv_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([h_min, s_min, v_min])
        upper_hsv = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)
        hsv_filtered = cv2.bitwise_and(current_frame, current_frame, mask=mask)

        self.publisher_.publish(self.br.cv2_to_imgmsg(hsv_filtered, encoding="bgr8"))
        cv2.imshow("HSV Filtered Image", hsv_filtered)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    hsv_filter_node = HSVFilterNode()
    rclpy.spin(hsv_filter_node)
    hsv_filter_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
