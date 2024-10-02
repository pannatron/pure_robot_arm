#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import threading
from ultralytics import YOLO

class TopViewImageSubscriber(Node):

    def __init__(self):
        super().__init__('top_view_image_subscriber')
        # Subscription to top_view_image topic
        self.subscription = self.create_subscription(
            Image,
            '/top_view_image',  # Replace this with your actual topic
            self.listener_callback,
            1
        )
        self.bridge = CvBridge()

        # Publisher for AI detected image
        self.ai_detected_pub = self.create_publisher(Image, '/ai_detected_image', 1)

        # Load YOLO model
        self.model = YOLO("/home/borot/Desktop/Bittle_BehaviorTree/src/inverse_perspective_mapping/scripts/model_- 16 september 2024 16_55.pt")

        # Frame counter to reduce workload
        self.frame_count = 0

    def listener_callback(self, msg):
        self.frame_count += 1
        # Only process every 3 frames to reduce workload
        if self.frame_count % 3 != 0:
            return

        # Convert ROS Image message to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Use threading to perform inference in a separate thread
        threading.Thread(target=self.run_inference, args=(frame,)).start()

    def run_inference(self, frame):
        # Perform inference
        try:
            results = self.model(frame)  # Run inference on the frame directly
            self.process_result(frame, results)  # Process and display the result
        except Exception as e:
            self.get_logger().error(f"Error performing inference: {str(e)}")

    def process_result(self, frame, results):
        # Loop through the results and draw bounding boxes on the frame
        for result in results:
            boxes = result.boxes.xyxy  # Bounding box coordinates
            confidences = result.boxes.conf  # Confidences
            class_indices = result.boxes.cls  # Class indices

            for box, class_index, confidence in zip(boxes, class_indices, confidences):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                label_name = self.model.names[int(class_index)]  # Convert class index to int and get class name
                conf = float(confidence)
                
                if conf > 0.7:
                    # Draw rectangle and label on the image
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f'{label_name} ({conf:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Publish the detected image
        detected_image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.ai_detected_pub.publish(detected_image_msg)


def main(args=None):
    rclpy.init(args=args)

    top_view_image_subscriber = TopViewImageSubscriber()

    try:
        rclpy.spin(top_view_image_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        top_view_image_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

