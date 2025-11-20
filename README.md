## 1. Prerequisites
You need **Python 3.8+** installed.

---

## 2. Clone the Repository
### 📥 Clone the Repository
```bash
git clone https://github.com/rahuljangra396/VisionSense.git
```

---

## 3. Install Dependencies
The required packages are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

---

## 4. Download the YOLOv8 Model
The application uses the **large YOLOv8 model**.

Download here (from Ultralytics):  
https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt  

Place the file in the project root directory alongside `vision_sense_app.py`.

---

## 5. AI Explanation (Optional)
If you want the **AI Explanation** feature:

1. Create a `.env` file in the project folder.  
2. Add your API key:

```env
OPENAI_API_KEY=your_key_here
```

---

## 6. Install Tesseract OCR (Only for Text Extraction)
Tesseract is required for reading text from images.

### Windows
Install from:  
https://github.com/UB-Mannheim/tesseract/wiki

---

## 7. Run the Application
Start the Streamlit app:

```bash
streamlit run vision_sense_app.py
```

The app will open in your browser.

---

## 📌 Usage
### Upload or Capture:
Use the **Upload an Image** or **Open Webcam** button to load an image.

### Modes available:
- Object Detection
- OCR (Text Extraction)
- Face Detection
- AI Explanation *(optional)*

