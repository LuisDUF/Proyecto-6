# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import colors, Annotator
from ultralytics.data.augment import LetterBox

s_mode = 0   # 0: semantic, 1: instance
mostrar_puntos = True  # Cambia a False para ocultar puntos del esqueleto
mostrar_segmentacion = True  # Cambia a False para desactivar segmentación

cap = cv2.VideoCapture("Proyect - 06/videos/walking_01.mp4")

# Detectar si CUDA está disponible y usarlo
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Usando dispositivo: {device.upper()}")

# === CARGA DE DOS MODELOS ===
pose_model = YOLO("yolo11s-pose.pt").to(device)
seg_model = YOLO("yolo11n-seg.pt").to(device)
names = seg_model.names

# Video writer
w, h, fps = (int(cap.get(x)) for x in
             (cv2.CAP_PROP_FRAME_WIDTH,
              cv2.CAP_PROP_FRAME_HEIGHT,
              cv2.CAP_PROP_FPS))
vw = cv2.VideoWriter(f"results_{s_mode}.avi",
                     cv2.VideoWriter_fourcc(*"mp4v"),
                     fps,
                     (w, h))

while cap.isOpened():

    success, im0 = cap.read()
    if not success:
        break

    # === RESULTADOS DE SEGMENTACIÓN ===
    seg_results = seg_model.track(im0, persist=True, device=device)[0]
    annotator = Annotator(im0)

    if seg_results.boxes.id is not None:

        boxes = seg_results.boxes.xyxy.tolist()
        tids = seg_results.boxes.id.int().tolist()
        clss = seg_results.boxes.cls.cpu().tolist()
        masks = seg_results.masks

        if mostrar_segmentacion and masks is not None and masks.data is not None:
            img = LetterBox(masks.shape[1:])(image=annotator.result())
            im_gpu = (torch.as_tensor
                      (img, dtype=torch.float16,
                       device=masks.data.device)
                      .permute(2, 0, 1).flip(0)
                      .contiguous() / 255)

            annotator.masks(masks.data, colors=[
                colors(x, True)
                for x in (tids if s_mode==1 else clss)],
                            im_gpu=im_gpu)

        for b, t, c in zip(boxes, tids, clss):
            annotator.box_label(
                b,
                color=colors(t if s_mode==1 else c, True),
                label=names[c])

    # === RESULTADOS DE POSE ===
    pose_results = pose_model(im0, verbose=False, device=device)[0]

    if mostrar_puntos and pose_results.keypoints is not None and len(pose_results.keypoints.xy) > 0:
        for kp in pose_results.keypoints.xy:
            keypoints = kp.cpu().numpy()
            for x, y in keypoints:
                cv2.circle(im0, (int(x), int(y)), 4, (0, 255, 0), -1)

    cv2.imshow("YOLOv8 Pose + Segmentation", im0)
    vw.write(im0)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
