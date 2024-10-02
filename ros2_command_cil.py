#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import serial
import time
from std_msgs.msg import String

class SimpleSerialCommander(Node):
    def __init__(self):
        super().__init__('simple_serial_commander')

        # Initialize serial connection
        self.ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)  # Adjust the port and timeout as needed
        time.sleep(5)  # Wait for the device to initialize

        self.get_logger().info("Serial connection established")

        # Subscription to ROS topic to receive commands
        self.command_subscriber = self.create_subscription(
            String,
            'serial_commands',
            self.command_callback,
            10
        )

        self.get_logger().info("Simple Serial Commander Node started, waiting for commands...")

    def send_command_and_receive_response(self, command):
        self.ser.flushInput()  # Flush the input buffer
        self.ser.flushOutput()  # Flush the output buffer
        self.ser.write(command.encode())  # Send command to the device

        # Try to read the response (non-blocking)
        try:
            response = self.ser.readline().decode().strip()  # Read response from the device
        except serial.SerialTimeoutException:
            response = "No response (timeout)"
        return response

    def command_callback(self, msg):
        command = msg.data
        self.get_logger().info(f"Sending command: {command.strip()}")
        response = self.send_command_and_receive_response(command)
        self.get_logger().info(f"Response: {response}")

    def destroy_node(self):
        self.ser.close()  # Close the serial connection
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SimpleSerialCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

