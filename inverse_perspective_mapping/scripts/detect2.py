import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import inference

class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/v4l/camera/image_raw',
            self.listener_callback,
            10)

        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.pred_x = None
        self.pred_y = None
        
        # Publishers for x and y coordinates in centimeters
        self.x_pub = self.create_publisher(Float32, 'x_coordinate_cm', 10)
        self.y_pub = self.create_publisher(Float32, 'y_coordinate_cm', 10)

        # Fetch the Roboflow model using the environment variable
        api_key = os.getenv('Oc4L94BZ2PztfQqdPz7U')
        if not api_key:
            self.get_logger().error('ROBOFLOW_API_KEY environment variable is not set')
            raise RuntimeError('ROBOFLOW_API_KEY environment variable is not set')
        
        self.model = inference.get_model("pure-s8xim/5", api_key)

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        result = self.perform_inference(frame)
        
        if result is not None:
            print("Inference result:", result)  # Debugging information
            # Check if result contains 'predictions' key
            if 'predictions' in result:
                self.pred_x = result['predictions']['x']
                self.pred_y = result['predictions']['y']
                # Draw bounding boxes on the frame
                for prediction in result['predictions']:
                    x1 = int(prediction['x'] - prediction['width'] / 2)
                    y1 = int(prediction['y'] - prediction['height'] / 2)
                    x2 = int(prediction['x'] + prediction['width'] / 2)
                    y2 = int(prediction['y'] + prediction['height'] / 2)
                    label = prediction['class']
                    confidence = prediction['confidence']
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Draw label and confidence
                    cv2.putText(frame, f'{label} ({confidence:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Display the result
                cv2.imshow("detect", frame)
                cv2.waitKey(1)
            else:
                print("No predictions found in the result")

    def perform_inference(self, frame):
        # Save frame to a temporary file
        temp_image_path = '/tmp/temp_frame.jpg'
        cv2.imwrite(temp_image_path, frame)
        
        # Perform inference on the saved image
        result = self.model.infer(image=temp_image_path)
        
        return result

def main(args=None):
    rclpy.init(args=args)

    image_subscriber = ImageSubscriber()

    rclpy.spin(image_subscriber)

    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
