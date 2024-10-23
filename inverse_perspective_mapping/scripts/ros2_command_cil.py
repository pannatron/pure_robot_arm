#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import serial
import time
from std_msgs.msg import String
from std_srvs.srv import Trigger

class SimpleSerialCommander(Node):
    def __init__(self, test_mode=False):
        super().__init__('simple_serial_commander')

        self.test_mode = False
        # Initialize serial connection only if not in test mode
        if not self.test_mode:
            self.ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)  # Adjust the port and timeout as needed
            time.sleep(5)  # Wait for the device to initialize
            self.get_logger().info("Serial connection established")

        self.command_buffer = None  # To store the latest AI command

        # Subscription to ROS topic to receive commands (for Manual UI commands)
        self.command_subscriber_manual = self.create_subscription(
            String,
            'serial_commands',
            self.command_callback_manual,
            10
        )

        # Subscription to ROS topic to receive commands (for AI commands)
        self.command_subscriber_ai = self.create_subscription(
            String,
            '/serial_commands_ai',
            self.command_callback_ai,
            10
        )

        # Service to trigger sending command to microcontroller (AI)
        self.srv = self.create_service(
            Trigger,
            'send_serial_command',
            self.trigger_command_callback
        )

        self.get_logger().info("Simple Serial Commander Node started, waiting for commands...")

    def command_callback_manual(self, msg):
        # Handle command from Manual UI and send immediately without waiting for response
        command = msg.data
        self.get_logger().info(f"Manual Command received: {command.strip()}")
        if self.test_mode:
            self.simulate_response(command)
        else:
            self.send_command(command)

    def command_callback_ai(self, msg):
        # Store the latest command in the buffer for AI
        self.command_buffer = msg.data
        self.get_logger().info(f"AI Command received and buffered: {self.command_buffer.strip()}")

    def send_command(self, command):
        if not self.test_mode:
            self.ser.flushInput()  # Flush the input buffer
            self.ser.flushOutput()  # Flush the output buffer
            self.ser.write(command.encode())  # Send command to the device
        else:
            # Simulate sending the command in test mode
            self.get_logger().info(f"[TEST MODE] Simulated sending command: {command.strip()}")
            self.simulate_response(command)

    def send_command_and_receive_response(self, command):
        if not self.test_mode:
            self.send_command(command)

            # Try to read the response (non-blocking)
            try:
                response = self.ser.readline().decode().strip()  # Read response from the device
            except serial.SerialTimeoutException:
                response = "No response (timeout)"
            return response
        else:
            # Simulate sending and receiving in test mode
            self.get_logger().info(f"[TEST MODE] Simulating command execution: {command.strip()}")
            time.sleep(5)  # Simulate delay as if performing the action
            return f"[TEST MODE] Command '{command.strip()}' executed successfully"

    def trigger_command_callback(self, request, response):
        if self.command_buffer:
            self.get_logger().info(f"Sending buffered AI command: {self.command_buffer.strip()}")
            serial_response = self.send_command_and_receive_response(self.command_buffer)

            if "successfully" in serial_response:
                self.get_logger().info(f"Response: {serial_response}")
                response.success = True
                response.message = f"Command executed successfully: {serial_response}"
            else:
                self.get_logger().warning("No response received from device.")
                response.success = False
                response.message = "Failed to receive response from device."
            
            # Clear the buffer after sending the command
            self.command_buffer = None
        else:
            response.success = False
            response.message = "No command buffered to send."

        return response

    def simulate_response(self, command):
        # Simulate a delay to mimic command execution time
        time.sleep(5)
        self.get_logger().info(f"[TEST MODE] Simulated response: Command '{command.strip()}' executed successfully")

    def destroy_node(self):
        if not self.test_mode:
            self.ser.close()  # Close the serial connection
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)

    # Set test_mode to True if you want to run in test mode
    test_mode = True
    node = SimpleSerialCommander(test_mode=test_mode)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
