import cv2
import numpy as np
import tempfile
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from ultralytics import YOLO
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import uvicorn
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Initialize the YOLO model once
model = YOLO('yolov8m_trained.pt')

# Write app description
description = """
    Model: 'yolov8m_trained.pt'\n
    Dataset: African wildlife dataset\n
    Upload an image to get predictions
"""

# app intialization
app = FastAPI(
    title="YoloV8 africain wildlife",
    description=description,
    summary="The goal of this app is to perform real-time animal detection in nature reserves.",
    version="0.0.1",
    contact={
        "name": "OUEDRAOGO FAOUZI BRICE PhD",
        "github": "https://github.com/HbkFaouzi",
        "linkedin": "https://www.linkedin.com/in/faouzi-brice-ouedraogo-ph-d-879671236/",
    }
    )

class PredictionResult(BaseModel):
    image: str

def predict_and_plot(image: np.ndarray) -> np.ndarray:
    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_image_file:
        image_path = temp_image_file.name
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Predict 
        res = model(image_path)
        res_plotted = res[0].plot()

        # Convert the plotted result back to RGB format
        res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        # Plot and save the result to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png") as temp_plot_file:
            plot_path = temp_plot_file.name
            plt.imshow(res_plotted_rgb)
            plt.axis('off')
            plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            # Read and return the plot
            plot_image = cv2.imread(plot_path)
            plot_image_rgb = cv2.cvtColor(plot_image, cv2.COLOR_BGR2RGB)
            return plot_image_rgb

@app.post("/predict/", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Predict and plot
    result_image = predict_and_plot(image_rgb)
    
    # Convert the result image to base64
    _, buffer = cv2.imencode('.png', result_image)
    result_image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return PredictionResult(image=result_image_base64)

# Running the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
