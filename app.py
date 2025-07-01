import os
import sys
import cv2
import numpy as np
import torch
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Add YOLO path
sys.path.append(os.path.join(os.getcwd(), 'yolov8'))
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
from utils.torch_utils import select_device
from utils.plots import Annotator

# Fix path for PyTorch on Windows
import pathlib
if sys.platform == 'win32':
    pathlib.PosixPath = pathlib.WindowsPath

# Load YOLO model
device = select_device('cpu')
yolo_model = DetectMultiBackend('best.pt', device=device)
yolo_model.warmup(imgsz=(1, 3, 640, 640))
stride, names, pt = yolo_model.stride, yolo_model.names, yolo_model.pt

# Load CNN model
cnn_model = tf.keras.models.load_model('cat_breed_classifier_face.h5')
CNN_CLASS_NAMES = [
    'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British Shorthair',
    'Egyptian_Mau', 'Havana', 'Maine Coon', 'Persian', 'Ragdoll',
    'Russian_Blue', 'Siamese', 'Sphynx'
]

# Load cat face Haar cascade
cat_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')


def run_yolo_detection(image_path):
    img0 = cv2.imread(image_path)
    img = letterbox(img0, 640, stride=stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(device).float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    pred = yolo_model(img_tensor, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    detections = []
    annotator = Annotator(img0.copy(), line_width=2)
    cropped_face = None

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img0.shape).round()
            for i, (*xyxy, conf, cls) in enumerate(reversed(det)):
                class_id = int(cls)
                label = f'{names[class_id]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=(255, 0, 0))
                detections.append({
                    "label": names[class_id],
                    "confidence": round(float(conf) * 100, 2)
                })
                if i == 0:
                    x1, y1, x2, y2 = map(int, xyxy)
                    cropped_face = img0[y1:y2, x1:x2]

    result_img = annotator.result()
    return result_img, detections, cropped_face


def run_cnn_prediction(cropped_img):
    if cropped_img is None:
        return "No face for CNN model"

    cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(cropped_rgb, (224, 224)).astype(np.float32) / 255.0
    x = np.expand_dims(resized, axis=0)

    preds = cnn_model.predict(x)[0]
    idx = np.argmax(preds)
    label = f"{CNN_CLASS_NAMES[idx]} ({preds[idx]*100:.2f}%)"
    return label


@app.route('/', methods=['GET', 'POST'])
def index():
    result_image = None
    detections = []
    cnn_result = None

    if request.method == 'POST':
        file = request.files.get('image')
        if not file:
            return render_template('index.html', error="No file uploaded")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result_img, detections, face_img = run_yolo_detection(filepath)

        # Save image
        result_filename = filename
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        cv2.imwrite(result_path, result_img)
        result_image = result_filename

        # CNN prediction
        if face_img is not None:
            cnn_result = run_cnn_prediction(face_img)
        else:
            cnn_result = "No cat face found for CNN model"

    return render_template("index.html", result_image=result_image,
                           detections=detections, cnn_result=cnn_result)


if __name__ == '__main__':
    app.run(debug=True)
