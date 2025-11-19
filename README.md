Markdown# 🧠 VisionSense — AI Image Analyzer

VisionSense is a versatile Streamlit application that combines several computer vision and AI capabilities to analyze images. It features **Object Detection** using a powerful **YOLOv8-large** model, **Optical Character Recognition (OCR)** with Tesseract, **Face Detection** using OpenCV, and an **AI Explanation** mode powered by OpenAI's language model.

## ✨ Features

* **Flexible Input:** Upload an image file (JPG, JPEG, PNG) or capture a new photo directly using your webcam.
* **Object Detection:** Identifies objects in the image using the `yolov8l.pt` model, with configurable confidence and IoU thresholds.
* **Text Extraction (OCR):** Extracts text from the image using **Pytesseract**.
* **Face Detection:** Locates human faces using an **OpenCV Haar Cascade** classifier.
* **AI Explanation (Optional):** Generates a concise, context-aware description and suggestions for the image by feeding the detected objects and OCR text to **OpenAI's GPT-3.5-turbo** (requires an API key).
* **Annotated Image Download:** Download the analyzed image with the relevant bounding boxes and labels.

## 🚀 Setup and Installation

Follow these steps to get VisionSense running on your local machine.

### 1. Prerequisites

You need **Python 3.8+** installed.

### 2. Clone the Repository

```bash

3. Install DependenciesThe required packages are listed in requirements.txt.Bashpip install -r requirements.txt
4. Download the YOLOv8 ModelThe application uses the large YOLOv8 model. You must download the yolov8l.pt file and place it in the project root directory alongside vision_sense_app.py.Download Link: yolov8l.pt (from Ultralytics)5. Configure OpenAI (Optional)If you wish to use the "AI Explanation" feature, you need an OpenAI API key.Create a file named .env in the project root folder.Add your API key to the file in the following format:Code snippetOPENAI_API_KEY="YOUR_API_KEY_HERE"
6. Install Tesseract OCRPytesseract is a wrapper for the Tesseract OCR engine, which must be installed separately on your system.Windows/macOS/Linux: Follow the installation instructions on the Tesseract GitHub page.💻 Running the ApplicationStart the Streamlit application from your terminal:Bashstreamlit run vision_sense_app.py
The application will open in your web browser.📝 UsageUpload or Capture: Use the "Upload an image" file uploader or the "Use camera (take photo)" button in the sidebar to load an image.Select Mode: Choose one of the four analysis modes from the sidebar radio buttons:Object DetectionText (OCR)Face DetectionAI ExplanationAdjust Settings (Object Detection/AI Explanation): If using a YOLO-based mode, adjust the Confidence threshold, NMS IoU threshold, and Max detections sliders/inputs in the sidebar to fine-tune the detection results.View Results: The results, including the annotated image (where applicable), dataframes, and text outputs, will appear in the main panel.Download: Use the "⬇ Download annotated image" link to save the processed image.⚙️ DependenciesThis project relies on the following key Python libraries:PackageDescriptionstreamlitThe framework for building the web application.ultralyticsFor high-performance object detection (YOLOv8).torchUnderlying deep learning library for YOLO.opencv-pythonUsed for image manipulation and face detection (Haar Cascade).pytesseractPython wrapper for Tesseract OCR.openaiUsed for the optional AI Explanation feature (GPT-3.5-turbo).pillow, numpy, pandasStandard libraries for image, numerical, and data handling.python-dotenvFor securely loading environment variables (OPENAI_API_KEY).
