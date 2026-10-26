import sys
import numpy as np
import cv2
import pydicom
import torch
import timm
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

# ---------------- константы (ансамбль, лучший результат на Kaggle) ----------------
CLASSIFIERS = [  # (имя timm, размер, веса)
    ("convnextv2_tiny.fcmae_ft_in22k_in1k", 512, "models/v1/classifier_best_for_f1.pt"),
    ("convnextv2_base.fcmae_ft_in22k_in1k_384", 640, "models/v2/classifier_best_for_f1.pt"),
]
YOLOS = [  # (размер, веса)
    (512, "models/v1/yolo_best.pt"),
    (640, "models/v2/yolo_best.pt"),
]
CLS_THRESHOLD = 0.45     # YOLO-боксы остаются, если средняя P(пневмония) >= порога
YOLO_CONF = 0.10         # минимальная уверенность бокса после объединения
WBF_IOU = 0.55           # с каким IoU боксы разных YOLO считаются одним очагом
K = 0.85                 # сжатие боксов к центру (тест Kaggle размечен более узкими боксами)

MEAN, STD = 0.449, 0.226
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["Normal", "Not Normal", "Pneumonia"]

# ---------------- модели ----------------
cls_models = []
for name, size, weights in CLASSIFIERS:
    m = timm.create_model(name, pretrained=False, num_classes=3, in_chans=1)
    m.load_state_dict(torch.load(weights, map_location=DEVICE))
    cls_models.append((m.to(DEVICE).eval(), size))
yolo_models = [(YOLO(weights), size) for size, weights in YOLOS]


# ---------------- препроцессинг, как в ноутбуке ----------------
def percentile_clip(data, lower_percentile=0.5, upper_percentile=99.5):
    low_val, high_val = np.percentile(data, [lower_percentile, upper_percentile])
    return np.clip(data, low_val, high_val)

def read_dicom(dicom_path):
    dcm = pydicom.dcmread(dicom_path)
    img = dcm.pixel_array.astype("float32")
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        img = img.max() - img
    img = percentile_clip(img)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + sys.float_info.epsilon)
    return np.round(img * 255).astype(np.uint8)

def resize_img(img, size=512):
    resized_img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return resized_img, resized_img.shape[0] / img.shape[0]


# ---------------- цепочка ----------------
def predict(dicom_path):
    img = read_dicom(dicom_path)
    h, w = img.shape

    # классификаторы: среднее вероятностей
    probs = []
    for m, size in cls_models:
        small, _ = resize_img(img, size)
        x = torch.from_numpy(small).float().div(255)[None, None]
        with torch.no_grad():
            probs.append(torch.softmax(m(((x - MEAN) / STD).to(DEVICE)), dim=1)[0].cpu().numpy())
    probs = np.mean(probs, axis=0)

    # YOLO, только если классификаторы видят пневмонию; боксы двух моделей объединяются через WBF
    boxes = []
    if probs[2] >= CLS_THRESHOLD:
        boxes_list, scores_list, labels_list = [], [], []
        for m, size in yolo_models:
            small, _ = resize_img(img, size)
            r = m.predict(np.repeat(small[..., None], 3, axis=2), imgsz=size, conf=0.01, verbose=False)[0]
            boxes_list.append(r.boxes.xyxyn.cpu().numpy())
            scores_list.append(r.boxes.conf.cpu().numpy())
            labels_list.append(np.zeros(len(r.boxes)))
        if sum(len(b) for b in boxes_list) > 0:
            xyxy, scores, _ = weighted_boxes_fusion(boxes_list, scores_list, labels_list,
                                                    iou_thr=WBF_IOU, skip_box_thr=0.01)
            for (x1, y1, x2, y2), c in zip(xyxy * [w, h, w, h], scores):
                if c >= YOLO_CONF:
                    bw, bh = x2 - x1, y2 - y1
                    boxes.append([x1 + bw * (1 - K) / 2, y1 + bh * (1 - K) / 2, bw * K, bh * K, c])

    # картинка с разметкой
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x1, y1, bw, bh, c in boxes:
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x1 + bw), int(y1 + bh)), (0, 0, 255), 3)
        cv2.putText(vis, f"{c:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    label = f"{CLASSES[probs.argmax()]}  P(pneumonia)={probs[2]:.2f}"
    cv2.putText(vis, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    out_path = str(dicom_path).replace(".dcm", "_pred.png")
    cv2.imwrite(out_path, vis)

    return probs, boxes, out_path


if __name__ == "__main__":
    probs, boxes, out_path = predict(input("Путь к .dcm: ").strip())
    print("вероятности [норма, не норма, пневмония]:", probs.round(3))
    print("боксы [x, y, w, h, conf]:", boxes if boxes else "нет")
    print("картинка:", out_path)
