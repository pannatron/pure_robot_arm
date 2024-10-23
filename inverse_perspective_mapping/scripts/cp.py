# import the inference-sdk
from inference_sdk import InferenceHTTPClient
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import inference

from roboflow import Roboflow

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Oc4L94BZ2PztfQqdPz7U"
)

cap = cv2.VideoCapture("/dev/video60")


while True:      
    if not cap.isOpened():
        print("Error: Could not open L4V2 camera.")
        break      
    
    # Capture frame-by-frame
    ret, frame = cap.read()
    result = CLIENT.infer(frame, model_id="pure-s8xim/5")

    # Display the frame

    print(result)
    cv2.imshow('L4V2 Camera', frame)

    # Wait for the 's' key to be pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Save the captured image
        image_filename = 'captured_image_l4v2.jpg'
        cv2.imwrite(image_filename, frame)
        print(f"Image saved as {image_filename}")
        # infer on a local image
        break
    elif key == ord('q'):
        # Quit the loop if 'q' is pressed
        break
        

cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_image()








results = model.poll_until_video_results(job_id)

print(results)
