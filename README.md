# 🐱 Cat Breed Detection and Classification (YOLOv8 + CNN + Flask)

A web-based application that combines **YOLOv8** for cat detection and **CNN (MobileNetV2)** for cat breed classification, built using **Flask**.

![demo](static/demo.png)

## 🚀 Features

- 📷 Upload cat images to the web interface  
- 🎯 Detect cats in the image using YOLOv8  
- 🧠 Classify cat breed using CNN (MobileNetV2) on the detected face  
- 📊 Show detection labels and confidence scores in a table  
- ❌ If no cat is detected, the app displays a warning  

---

## 🗂️ Project Structure

```
├── app.py                       # Main Flask app (YOLO + CNN integration)
├── yolov8/                     # YOLOv8 model directory (custom-trained model)
│   ├── models/
│   ├── utils/
│   └── ...
├── cat_breed_classifier_face.h5 # CNN model for classifying 13 cat breeds
├── best.pt                     # YOLOv8 PyTorch model (object detection)
├── static/
│   ├── uploads/                # Uploaded images
│   └── results/                # YOLO+CNN result images
├── templates/
│   └── index.html              # Frontend HTML (Bootstrap)
└── README.md
```

---

## 🐍 Requirements

Install Python dependencies using pip:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```txt
flask
opencv-python
torch
torchvision
tensorflow
numpy
Pillow
```

> 🔧 You also need the `yolov8` directory from [Ultralytics](https://github.com/ultralytics/yolov5) (or your own forked and trained version), placed in the project root.

---

## 🧠 Models Used

### 1. YOLOv8 (Object Detection)
- Trained on a custom dataset of cat breeds (e.g., Abyssinian, Siamese, Persian, etc.)
- Output: Bounding box + breed name + confidence

### 2. CNN with MobileNetV2 (Classification)
- Takes in the cropped face region from YOLO or Haar Cascade
- Output: Top-1 predicted breed with confidence score
- Supports 13 cat breeds:
  ```
  Abyssinian, Bengal, Birman, Bombay, British Shorthair, 
  Egyptian_Mau, Havana, Maine Coon, Persian, Ragdoll, 
  Russian_Blue, Siamese, Sphynx
  ```

---

## 🖥️ How to Run

1. **Clone this repo**
   ```bash
   git clone https://github.com/yourusername/cat-breed-detector.git
   cd cat-breed-detector
   ```

2. **Ensure models are in place**
   - `best.pt` – YOLOv8 PyTorch model
   - `cat_breed_classifier_face.h5` – CNN model
   - YOLO directory (`yolov8/`) is available

3. **Run the Flask App**
   ```bash
   python app.py
   ```

4. **Open browser**
   Visit: `http://127.0.0.1:5000`

---

## 📸 Example Usage

Upload a cat photo:
- ✅ If a cat is detected, the result shows:
  - Detected bounding box
  - YOLO label(s)
  - CNN breed prediction on detected face
- ❌ If no cat is detected, a warning message will be shown.

---

## ⚠️ Notes

- For **Windows users**, the script handles `pathlib.PosixPath` compatibility with PyTorch.
- Haar cascade (`haarcascade_frontalcatface.xml`) is used as a fallback to crop face if YOLO fails.
- You can improve YOLO detection by training on higher-resolution datasets and adding more breed variations.

---

## 📚 Citation

If you use this project or build on top of it, please cite the original YOLO and MobileNetV2 papers.

---
