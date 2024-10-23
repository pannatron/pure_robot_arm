#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String,Int32
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
import cv2
from ultralytics import YOLO
import time

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
        
                # Subscription to X and Y offsets
        self.x_offset_subscription = self.create_subscription(
            Int32,
            '/x_offset',
            self.x_offset_callback,
            1
        )

        self.y_offset_subscription = self.create_subscription(
            Int32,
            '/y_offset',
            self.y_offset_callback,
            1
        )
  # Subscription to delay times
        self.move_to_delay_sub = self.create_subscription(
            Int32,
            'move_to_delay',
            self.move_to_delay_callback,
            1
        )
        self.pick_delay_sub = self.create_subscription(
            Int32,
            'pick_delay',
            self.pick_delay_callback,
            1
        )
        self.place_delay_sub = self.create_subscription(
            Int32,
            'place_delay',
            self.place_delay_callback,
            1
        )

        # Subscription to Z value
        self.z_value_sub = self.create_subscription(
            Int32,
            'z_value',
            self.z_value_callback,
            1
        )

        # Default values (initialize to some default)
        self.move_to_delay = 10  # Default move delay (seconds)
        self.pick_delay = 5      # Default pick delay (seconds)
        self.place_delay = 5     # Default place delay (seconds)
        self.z_value = 30.0     # Default Z-axis value (mm)
        # Variables to store offset values
        self.x_offset = 0
        self.y_offset = 0
        self.width_cm = 70.0
        self.height_cm = 50.0
        self.width_px = 600
        self.height_px = 600
        self.scale_x = self.width_cm / self.width_px
        self.scale_y = self.height_cm / self.height_px

        self.bridge = CvBridge()

        # Publisher for AI detected image
        self.ai_detected_pub = self.create_publisher(Image, '/ai_detected_image', 1)

        # Publisher for action commands
        self.command_pub = self.create_publisher(String, 'serial_commands', 10)

        # Load YOLO model
# Load YOLO model with verbose set to False
        self.model = YOLO("/home/pure/Desktop/Bittle_BehaviorTree/src/inverse_perspective_mapping/scripts/model_- 16 september 2024 16_55.pt", verbose=False)

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
    def move_to_delay_callback(self, msg):
        self.move_to_delay = msg.data
        self.get_logger().info(f"Move to delay updated: {self.move_to_delay}s")

    def pick_delay_callback(self, msg):
        self.pick_delay = msg.data
        self.get_logger().info(f"Pick delay updated: {self.pick_delay}s")

    def place_delay_callback(self, msg):
        self.place_delay = msg.data
        self.get_logger().info(f"Place delay updated: {self.place_delay}s")

    def z_value_callback(self, msg):
        self.z_value = 30.00
        self.get_logger().info(f"Z-axis value updated: {self.z_value}mm")

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
            self.results = results  # Save the results for later use
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
    def x_offset_callback(self, msg):
        self.x_offset = msg.data
        self.get_logger().info(f"Received X offset: {self.x_offset}")

    def y_offset_callback(self, msg):
        self.y_offset = msg.data
        self.get_logger().info(f"Received Y offset: {self.y_offset}")

    def execute_action(self):
        # ตรวจสอบว่ามีการตรวจจับเหลืออยู่หรือไม่
        if self.detection_list and self.are_tray_positions_complete():
            label_name = self.detection_list[0]  # รับการตรวจจับแรกในรายการ

            if self.current_step == "move_to_target":
                object_position = self.get_object_position_from_detection(label_name, self.results)
                if object_position is not None:
                    x, y = object_position
                    adjusted_x = x + self.x_offset
                    adjusted_y = y + self.y_offset
                    # ใช้ Z value ที่แน่นอน
                    move_command = f"move_to>{adjusted_x:.2f},{adjusted_y:.2f},100.00,90.0>\\n"
                    self.command_pub.publish(String(data=move_command))
                    self.get_logger().info(f"Moving to object position with offsets: ({adjusted_x}, {adjusted_y}), Z: 100.00")

                    # หน่วงเวลา move delay
                    time.sleep(self.move_to_delay)

                    # ย้ายไปยังขั้นตอนถัดไป
                    self.current_step = "pick"
                    self.execute_action()  # เรียกฟังก์ชันเพื่อดำเนินการขั้นตอนถัดไป

            elif self.current_step == "pick":
                object_position = self.get_object_position_from_detection(label_name, self.results)
                if object_position is not None:
                    x, y = object_position
                    adjusted_x = x + self.x_offset
                    adjusted_y = y + self.y_offset

                    # เคลื่อนที่ไปยังตำแหน่งหยิบ
                    move_command = f"move_to>{adjusted_x:.2f},{adjusted_y:.2f},{self.z_value:.2f},90.0>\\n"
                    self.command_pub.publish(String(data=move_command))
                    self.get_logger().info(f"Moving to object position with offsets: ({adjusted_x}, {adjusted_y}), Z: {self.z_value}")

                    # หน่วงเวลา move delay
                    time.sleep(self.move_to_delay)

                    # ส่งคำสั่ง pick
                    pick_command = "pick>\\n"
                    self.command_pub.publish(String(data=pick_command))
                    self.get_logger().info(f"Pick command executed for {label_name}")

                    # หน่วงเวลา pick delay
                    time.sleep(self.pick_delay)

                    # ย้ายไปยังขั้นตอนถัดไป
                    self.current_step = "move_to_tray"
                    self.execute_action()

            elif self.current_step == "move_to_tray":
                # เคลื่อนที่ไปยังตำแหน่งถาด
                tray_position_key = self.get_tray_key_from_class_name(label_name)
                if tray_position_key and tray_position_key in self.tray_positions:
                    tray_position = self.tray_positions[tray_position_key]
                    if tray_position:
                        tray_x, tray_y = tray_position
                        move_to_tray_command = f"move_to>{tray_x:.2f},{tray_y:.2f},100.00,90.0>\\n"
                        self.command_pub.publish(String(data=move_to_tray_command))
                        self.get_logger().info(f"Moving to tray position: ({tray_x}, {tray_y}), Z: 100.00")

                        # หน่วงเวลา move delay
                        time.sleep(self.move_to_delay)

                        # ย้ายไปยังขั้นตอนถัดไป
                        self.current_step = "place"
                        self.execute_action()

            elif self.current_step == "place":
                # วางวัตถุ
                tray_position_key = self.get_tray_key_from_class_name(label_name)
                if tray_position_key and tray_position_key in self.tray_positions:
                    tray_position = self.tray_positions[tray_position_key]
                    if tray_position:
                        tray_x, tray_y = tray_position
                        move_to_tray_command = f"move_to>{tray_x:.2f},{tray_y:.2f},{self.z_value:.2f},90.0>\\n"
                        self.command_pub.publish(String(data=move_to_tray_command))
                        self.get_logger().info(f"Moving to tray position: ({tray_x}, {tray_y}), Z: {self.z_value}")

                        # หน่วงเวลา move delay
                        time.sleep(self.move_to_delay)

                    # ส่งคำสั่งวาง
                    place_command = "place>\\n"
                    self.command_pub.publish(String(data=place_command))
                    self.get_logger().info(f"Place command executed for {label_name} ✅")

                    # ลบ label ที่ประมวลผลเสร็จแล้วออกจากรายการ
                    self.detection_list.pop(0)

                    # ถ้าไม่มีการตรวจจับเหลืออยู่ ให้แสดง "ALL DONE"
                    if not self.detection_list:
                        self.get_logger().info("ALL DONE!! 🎉")

                    # รีเซ็ตขั้นตอนสำหรับการตรวจจับถัดไป
                    self.current_step = "move_to_target"
                    self.is_processing_command = False

                    # เรียกฟังก์ชันต่อเนื่องถ้ามีวัตถุอื่นๆ ในรายการ
                    if self.detection_list:
                        self.execute_action()


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
    def get_object_position_from_detection(self, label_name, results):
        # Loop through the detections and find the position of the object with the given label_name
        for result in results:
            boxes = result.boxes.xyxy  # Bounding box coordinates
            class_indices = result.boxes.cls  # Class indices
            
            for box, class_index in zip(boxes, class_indices):
                if self.model.names[int(class_index)] == label_name:
                    # Get the center of the bounding box
                    x_center = (box[0] + box[2]) / 2
                    y_center = (box[1] + box[3]) / 2
                    
                    # Convert the image coordinates to world coordinates (in cm)
                    cm_x = ((x_center - 300) * self.scale_x) * 10  # Adjust the 300 based on your camera setup
                    cm_y = (((y_center - 600) * self.scale_y) * -1) * 10  # Adjust the 600 based on your camera setup
                    
                    self.get_logger().info(f"Object {label_name} detected at image coordinates: ({x_center}, {y_center})")
                    self.get_logger().info(f"Converted to world coordinates: ({cm_x}, {cm_y})")
                    
                    return (cm_x, cm_y)
        
        return None  # Return None if the object is not found

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
