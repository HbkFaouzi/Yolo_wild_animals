# Yolo Wildlife Detection

This project uses the YOLO (You Only Look Once) model for real-time animal detection in nature reserves. The model is trained on an African wildlife dataset.

## Project Structure
```
stock-market-sentiment-analysis

├── training
|   └── yolo_wildlife.ipynb
├── api
|   └── app.py
|   └── gradio_app.py
|   └── test.py
├── README.md

```

## Key Files

- [`app.py`](command:_github.copilot.openSymbolInFile?%5B%22app.py%22%2C%22app.py%22%5D "app.py"): This is the main application file. It initializes the YOLO model and sets up a FastAPI server for handling image upload and prediction requests.

- [`gradio_app.py`](command:_github.copilot.openSymbolInFile?%5B%22gradio_app.py%22%2C%22gradio_app.py%22%5D "gradio_app.py"): This file sets up a Gradio interface for the application. It allows users to upload images and view the prediction results.

- [`test.py`](command:_github.copilot.openSymbolInFile?%5B%22test.py%22%2C%22test.py%22%5D "test.py"): This script tests the prediction endpoint of the FastAPI server.

- [`yolo_wildlife.ipynb`](command:_github.copilot.openSymbolInFile?%5B%22yolo_wildlife.ipynb%22%2C%22yolo_wildlife.ipynb%22%5D "yolo_wildlife.ipynb"): This Jupyter notebook contains the code for training the YOLO model.

- [`yolov8m_trained.pt`](command:_github.copilot.openSymbolInFile?%5B%22yolov8m_trained.pt%22%2C%22yolov8m_trained.pt%22%5D "yolov8m_trained.pt"): This is the trained YOLO model.

## Usage

* To run the FastAPI server, execute the following command:

```bash
python api/app.py
```

* To launch the Gradio interface, execute the following command:
```bash
python api/gradio_app.py
```

* To test the prediction endpoint of the FastAPI server, execute the following command:
```bash
python api/test.py
```

## Contact

For any questions or concerns, please reach out to [OUEDRAOGO FAOUZI BRICE PhD](https://www.linkedin.com/in/faouzi-brice-ouedraogo-ph-d-879671236/). You can also check out the project on [GitHub](https://github.com/HbkFaouzi).
