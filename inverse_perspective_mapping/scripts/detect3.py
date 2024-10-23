import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import base64
import requests


class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/v4l/camera/image_raw',
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

        # Set your API key and model ID
        self.api_key = "qKDToPD1uyjiNxD4So1X"  # Replace with your actual API key
        self.model_id = "shape-i5uau/2"
        self.api_url = f"https://detect.roboflow.com/{self.model_id}/predict"

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        result = self.perform_inference(frame)

        if result is not None:
            print("Inference result:", result)  # Debugging information
            if 'predictions' in result:
                # Handle predictions
                for prediction in result['predictions']:
                    x1 = int(prediction['x'] - prediction['width'] / 2)
                    y1 = int(prediction['y'] - prediction['height'] / 2)
                    x2 = int(prediction['x'] + prediction['width'] / 2)
                    y2 = int(prediction['y'] + prediction['height'] / 2)
                    label = prediction['class']
                    confidence = prediction['confidence']

                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} ({confidence:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Display the result
                cv2.imshow("detect", frame)
                cv2.waitKey(1)
            else:
                print("No predictions found in the result")

    def perform_inference(self, frame):
        # Encode the image to JPEG format and then to base64
        _, buffer = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Prepare the payload
        payload = {
            "image": image_base64
        }
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        # Make the API request
        response = requests.post(self.api_url, headers=headers, json=payload)

        # Check for a successful response
        if response.status_code == 200:
            return response.json()  # Return the JSON response
        else:
            self.get_logger().error(f"Error: {response.status_code} - {response.text}")
            return None


def main(args=None):
    rclpy.init(args=args)

    image_subscriber = ImageSubscriber()

    rclpy.spin(image_subscriber)

    image_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
