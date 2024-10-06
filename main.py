import cv2
import numpy as np
import torch
from models import HybridModel
from models.experimental import attempt_load
from torchvision.transforms import functional as F
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box

# Load the pre-trained hybrid model
model = HybridModel()
model.load_state_dict(torch.load('hybrid_model.pth'))
model.eval()

# Load the YOLOv5 model
yolo_model = attempt_load('yolov5s.pt', map_location=torch.device('cpu'))

# Initialize camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform face detection using YOLOv5
    img = letterbox(frame, new_shape=640)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(torch.float32).unsqueeze(0) / 255.0
    pred = yolo_model(img, augment=False)[0]

    # Apply non-maximum suppression
    pred = non_max_suppression(pred, 0.4, 0.5)[0]

    # Extract bounding box coordinates
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()

        # Predict whether the detected faces are real or fake
        for *xyxy, conf, cls in pred:
            xyxy = [int(i) for i in xyxy]
            face = frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = F.normalize(F.to_tensor(face), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            face = face.unsqueeze(0)
            with torch.no_grad():
                output = model(face)
                _, predicted = torch.max(output, 1)
                confidence = torch.softmax(output, dim=1)[0][predicted.item()].item()
                label = 'Real' if predicted.item() == 0 else 'Fake'
                color = (0, 255, 0) if predicted.item() == 0 else (0, 0, 255)
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
                cv2.putText(frame, f'{label}: {confidence:.2f}', (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
