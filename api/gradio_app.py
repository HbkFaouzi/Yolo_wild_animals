import gradio as gr
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO

# Initialize the YOLO model once
model = YOLO('yolov8m_trained.pt')
# Write app description
description = """
    model: 'yolov8m_trained.pt'\n
    dataset: African wildlife dataset\n
    Upload an image to get predictions
"""

def predict_and_plot(image):
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

# Define the Gradio interface
input_image = gr.Image(type="numpy")
output_image = gr.Image(type="numpy")

interface = gr.Interface(fn=predict_and_plot, inputs=input_image, outputs=output_image, 
                         title="Image Prediction", description=description)

# Launch the interface
interface.launch()