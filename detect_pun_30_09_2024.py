#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
import cv2
from ultralytics import YOLO

class TopViewImageSubscriber(Node):

    def __init__(self):
        super().__init__('top_view_image_subscriber')
        
        # Subscription to top_view_image topic
        self.subscription = self.create_subscription(
            Image,
            '/top_view_image',
            self.listener_callback,
            1
        )
        
        # Subscription to tray_positions topic
        self.tray_subscription = self.create_subscription(
            String,
            '/tray_positions',
            self.tray_position_callback,
            1
        )

        # Subscription to ai_start_process topic
        self.ai_start_subscription = self.create_subscription(
            String,
            '/ai_start_process',
            self.ai_start_callback,
            1
        )
        
        self.bridge = CvBridge()

        # Publisher for AI detected image
        self.ai_detected_pub = self.create_publisher(Image, '/ai_detected_image', 1)

        # Publisher for action commands
        self.command_pub = self.create_publisher(String, 'serial_commands_ai', 10)

        # Load YOLO model
# Load YOLO model with verbose set to False
        self.model = YOLO("/home/borot/Desktop/Bittle_BehaviorTree/src/inverse_perspective_mapping/scripts/model_- 16 september 2024 16_55.pt", verbose=False)

        # Frame counter to reduce workload
        self.frame_count = 0

        # Dictionary to store tray positions
        self.tray_positions = {
            "circle_large": None,
            "circle_medium": None,
            "circle_small": None,
            "square_large": None,
            "square_medium": None,
            "square_small": None,
            "triangle_large": None,
            "triangle_medium": None,
            "triangle_small": None,
        }

        # State to track AI process and action sending
        self.ai_process_active = False
        self.is_processing_command = False
        self.detection_list = []
        self.current_step = "move_to_target"  # Tracks which step of the action is being processed

    def listener_callback(self, msg):
        # AI works regardless of AI ON or OFF state, just don't send commands if it's OFF
        self.frame_count += 1
        # Only process every 3 frames to reduce workload
        if self.frame_count % 3 != 0 or self.is_processing_command:
            return

        # Convert ROS Image message to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Perform inference
        self.run_inference(frame)

    def ai_start_callback(self, msg):
        if msg.data == "ON":
            self.ai_process_active = True
            self.get_logger().info("AI process started.")
        elif msg.data == "OFF":
            self.ai_process_active = False
            self.get_logger().info("AI process stopped.")

    def run_inference(self, frame):
        # Perform inference
        try:
            results = self.model(frame)  # Run inference on the frame directly
            self.process_result(frame, results)  # Process and display the result
        except Exception as e:
            self.get_logger().error(f"Error performing inference: {str(e)}")

    def process_result(self, frame, results):
        # If AI is not active or tray positions are not complete, skip sending commands
        if not self.ai_process_active or not self.are_tray_positions_complete():
            self.get_logger().info("AI is not active or tray positions are not complete, skipping command execution.")

        for result in results:
            boxes = result.boxes.xyxy  # Bounding box coordinates
            confidences = result.boxes.conf  # Confidences
            class_indices = result.boxes.cls  # Class indices

            for box, class_index, confidence in zip(boxes, class_indices, confidences):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                label_name = self.model.names[int(class_index)]  # Convert class index to int and get class name
                conf = float(confidence)

                # Skip if the detected object is 'Target'
                if label_name == "Target":
                    continue
                    
                if conf > 0.7:
                    # Draw rectangle and label on the image
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f'{label_name} ({conf:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                    if  self.ai_process_active or  self.are_tray_positions_complete:
                        # Add detected object to detection list if it's not already there
                        if label_name not in self.detection_list:
                            self.detection_list.append(label_name)

        # Publish the detected image
        detected_image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.ai_detected_pub.publish(detected_image_msg)

        # Execute action if there are detections and AI is active
        if self.ai_process_active and not self.is_processing_command and self.detection_list:
            self.is_processing_command = True
            self.execute_action()


    def execute_action(self):
        # Only execute action if there are detections in the list
        if self.detection_list and self.are_tray_positions_complete():
            label_name = self.detection_list[0]  # Get the first detection without removing it from the list
            tray_position_key = self.get_tray_key_from_class_name(label_name)

            if tray_position_key is not None and tray_position_key in self.tray_positions:
                tray_position = self.tray_positions[tray_position_key]

                if tray_position is not None:
                    if self.current_step == "move_to_target":
                        # Publish move_to command to go to the target
                        x, y = tray_position
                        move_command = f"move_to>{x:.2f},{y:.2f},120.0,90.0>\\n"
                        self.command_pub.publish(String(data=move_command))
                        self.get_logger().info(f"Moving to target position: ({x}, {y})")

                        # Wait for response before continuing
                        self.current_step = "pick"
                        self.wait_for_response()

                    elif self.current_step == "pick":
                        # Publish pick command once at the target
                        pick_command = "pick>\\n"
                        self.command_pub.publish(String(data=pick_command))
                        self.get_logger().info(f"Pick command executed for {label_name}")

                        # Wait for response before continuing
                        self.current_step = "move_to_tray"
                        self.wait_for_response()

                    elif self.current_step == "move_to_tray":
                        # Move to tray position using the stored tray position for this type of object
                        tray_x, tray_y = tray_position
                        move_to_tray_command = f"move_to>{tray_x:.2f},{tray_y:.2f},120.0,90.0>\\n"
                        self.command_pub.publish(String(data=move_to_tray_command))
                        self.get_logger().info(f"Moving to tray position: ({tray_x}, {tray_y})")

                        # Wait for response before continuing
                        self.current_step = "place"
                        self.wait_for_response()

                    elif self.current_step == "place":
                        # Move to tray position and place the object
                        place_command = "place>\\n"
                        self.command_pub.publish(String(data=place_command))
                        self.get_logger().info(f"Place command executed for {label_name} ✅")

                        # Remove the processed label from the list
                        self.detection_list.pop(0)

                        # If no more detections left, print ALL DONE
                        if not self.detection_list:
                            self.get_logger().info("ALL DONE!! 🎉")

                        # Reset the current step to move to next target
                        self.current_step = "move_to_target"
                        self.is_processing_command = False

                        # Add a separator line for readability
                        self.get_logger().info("----------")

    def get_tray_key_from_class_name(self, class_name):
        class_mapping = {
            "Cylinder_L": "circle_large",
            "Cylinder_M": "circle_medium",
            "Cylinder_S": "circle_small",
            "Square_L": "square_large",
            "Square_M": "square_medium",
            "Square_S": "square_small",
            "Triangle_L": "triangle_large",
            "Triangle_M": "triangle_medium",
            "Triangle_S": "triangle_small",
        }

        return class_mapping.get(class_name)

    def are_tray_positions_complete(self):
        # Check if all tray positions have been set
        for key, value in self.tray_positions.items():
            if value is None:
                return False
        return True

    def tray_position_callback(self, msg):
        # Example input: 'circle_large>171.50,73.33'
        data = msg.data.strip()
        try:
            tray_type, position = data.split(">")
            x, y = position.split(",")
            x, y = float(x), float(y)

            if tray_type in self.tray_positions:
                self.tray_positions[tray_type] = (x, y)
                self.get_logger().info(f"Updated position for {tray_type}: X={x}, Y={y}")
            else:
                self.get_logger().warning(f"Unknown tray type received: {tray_type}")
        except ValueError:
            self.get_logger().error(f"Invalid data format received: {data}")

    def wait_for_response(self):
        # Create a client for Trigger service to wait for response from the microcontroller
        client = self.create_client(Trigger, 'send_serial_command')
        
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for send_serial_command service...')
        
        request = Trigger.Request()
        future = client.call_async(request)

        # Use a callback to handle the response
        future.add_done_callback(self.response_callback)

    def response_callback(self, future):
        try:
            result = future.result()
            if result.success:
                self.get_logger().info(f"Response: {result.message}")
            else:
                self.get_logger().warning(f"Failed to execute command: {result.message}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {str(e)}")
        
        # Allow next step to proceed after the current one is completed
        self.is_processing_command = False

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
