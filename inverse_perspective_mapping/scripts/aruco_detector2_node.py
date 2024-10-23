#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String
from cv_bridge import CvBridge
import cv2
import numpy as np

class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        self.subscription = self.create_subscription(
            Image,
            '/v4l/camera/image_raw',
            self.image_callback,
            1)
        self.br = CvBridge()
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.kalman_filters = {}
        self.last_detected_corners = {}
        self.last_detected_directions = {}
        self.locked = False
        self.lock_ellipse = None
        self.lock_danger_ellipse = None

        self.clicked_points = []
        self.clicked_point = None
        self.width_cm =  70.0  # ความกว้างของพื้นที่ในหน่วยเซนติเมตร
        self.height_cm = 50.0  # ความสูงของพื้นที่ในหน่วยเซนติเมตร
        self.width_px =  600  # ความกว้างของภาพในหน่วยพิกเซล
        self.height_px = 600  # ความสูงของภาพในหน่วยพิกเซล
        self.scale_x = self.width_cm / self.width_px
        self.scale_y = self.height_cm / self.height_px

        # Publishers for x and y coordinates in centimeters
        self.x_pub = self.create_publisher(Float32, 'x_coordinate_cm', 10)
        self.y_pub = self.create_publisher(Float32, 'y_coordinate_cm', 10)
        # Publisher for serial commands
        self.command_pub = self.create_publisher(String, 'serial_commands', 10)

        self.count = 672
        cv2.namedWindow('Image')    
        cv2.setMouseCallback('Image', self.mouse_click)
        cv2.namedWindow('Top-View Perspective')
        cv2.setMouseCallback('Top-View Perspective', self.mouse_click_topview)

        self.get_logger().info('Aruco Detector Node has been started.')

    def nothing(self, x):
        pass

    def toggle_lock(self, x):
        self.locked = bool(x)
        if not self.locked:
            self.lock_ellipse = None
            self.lock_danger_ellipse = None

    def mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_points.append((x, y))
            if len(self.clicked_points) == 4:
                self.process_homography()

    def mouse_click_topview(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and hasattr(self, 'h_matrix'):
            self.clicked_point = (x, y)
            self.send_command_to_move(self.clicked_point)

    def send_command_to_move(self, clicked_point):
        cm_x = (clicked_point[0] - 300) * self.scale_x
        cm_y = (clicked_point[1] - 600) * self.scale_y
        command_str = f"move_to>{cm_x:.2f},{cm_y:.2f},0.0,90.0>\n"
        self.command_pub.publish(String(data=command_str))
        self.get_logger().info(f'Sent command: {command_str}')

    def process_homography(self):
        if len(self.clicked_points) == 4:
            pts_src = np.array(self.clicked_points, dtype="float32")
            pts_dst = np.array([
                [0, 0],
                [self.width_px - 1, 0],
                [self.width_px - 1, self.height_px - 1],
                [0, self.height_px - 1]
            ], dtype="float32")

            self.h_matrix, _ = cv2.findHomography(pts_src, pts_dst)
            self.clicked_points = []

    def image_callback(self, msg):
        frame = self.br.imgmsg_to_cv2(msg, 'bgr8')
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, rejectedCandidates = detector.detectMarkers(gray)

        if ids is not None:
            for corner in corners:
                cv2.polylines(frame, [corner.astype(np.int32)], True, (0, 255, 0), 2)

        if hasattr(self, 'h_matrix'):
            top_view = cv2.warpPerspective(frame, self.h_matrix, (self.width_px, self.height_px))
            
            if self.clicked_point:
                cm_x = (self.clicked_point[0] - 300) * self.scale_x
                cm_y = (self.clicked_point[1] - 600) * self.scale_y
                text = f"({cm_x:.2f} cm, {cm_y:.2f} cm)"
                cv2.circle(top_view, self.clicked_point, 5, (0, 0, 255), -1)
                cv2.putText(top_view, text, (self.clicked_point[0] + 10, self.clicked_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                # Publish the x and y coordinates in centimeters
                self.x_pub.publish(Float32(data=cm_x))
                self.y_pub.publish(Float32(data=cm_y))

            cv2.imshow('Top-View Perspective', top_view)

            # Check for space bar press to save the image
            if cv2.waitKey(1) & 0xFF == ord(' '):
                cv2.imwrite(f'image/{self.count}_top_view_image.jpg', top_view)
                self.get_logger().info(f'Top-view image saved as image/{self.count}_top_view_image.jpg')
                self.count += 1

        for point in self.clicked_points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"({point[0]}, {point[1]})", (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('Image', frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    aruco_detector = ArucoDetector()
    rclpy.spin(aruco_detector)

    aruco_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
