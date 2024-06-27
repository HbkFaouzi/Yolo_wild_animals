import requests
from PIL import Image
from io import BytesIO
import base64

# Path to the input image
image_path = 'buffalo_273.jpg'

# Make the request to the FastAPI server
with open(image_path, 'rb') as img_file:
    response = requests.post("http://127.0.0.1:8000/predict/", files={"file": img_file})

# Check if the request was successful
if response.status_code == 200:
    result = response.json()
    # Decode the base64-encoded image
    image_data = base64.b64decode(result['image'])
    # Save the image to a file
    output_path = "result_image.png"
    with open(output_path, "wb") as f:
        f.write(image_data)
    print(f"Image saved to {output_path}")
else:
    print("Request failed with status code:", response.status_code)
