import json
import requests

# Run inference on an image
url = "https://predict.ultralytics.com"
headers = {"x-api-key": "2106d288883771d815e4e61d7ee866bbaf250d5313"}
data = {"model": "https://hub.ultralytics.com/models/pHwCIjHjmaWieofisrxJ", "imgsz": 640, "conf": 0.25, "iou": 0.45}
with open("top_view_image.jpg", "rb") as f:
	response = requests.post(url, headers=headers, data=data, files={"file": f})

# Check for successful response
response.raise_for_status()

# Print inference results
print(json.dumps(response.json(), indent=2))

