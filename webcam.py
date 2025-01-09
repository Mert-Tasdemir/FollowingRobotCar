from ultralytics import YOLO
from ultralytics import settings
import cv2
import numpy as np
import math

MIN_CONFIDENCE: float = 0.54
MAX_SCOPE: float = 0.3
MAX_SPEED: float = 100.0
MAX_SPEEDS_LEN: int = 10

def draw_speedometer(img, x, speed):
    """
    Рисует спидометр на изображении.

    img: Исходное изображение.
    speed: Текущая скорость (0-100).
    """
    height, width, _ = img.shape
    line_width = 2
    radius = 100  # Радиус окружности спидометра
    center = (x - line_width//2, height - radius - line_width//2 - 20)  # Центр спидометра
    d_angle = -2.45
    d_width = 1.45

    speed = (1.0 - speed) * MAX_SPEED

    # Нарисовать круг
    cv2.circle(img, center, radius, (255, 255, 255), line_width)

    # Отметки на шкале (каждые 10 единиц)
    for i in range(0, 101, 10):
        angle = d_angle * math.pi / 2 + (math.pi * i / MAX_SPEED * d_width)
        x1 = int(center[0] + radius * math.cos(angle))
        y1 = int(center[1] + radius * math.sin(angle))
        x2 = int(center[0] + (radius - 10) * math.cos(angle))
        y2 = int(center[1] + (radius - 10) * math.sin(angle))
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), line_width)

        # Подписать числа на шкале
        label_x = int(center[0] + (radius - 25) * math.cos(angle))
        label_y = int(center[1] + (radius - 25) * math.sin(angle))
        cv2.putText(img, str(i), (label_x - 10, label_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (255, 255, 255), 1)

    # Нарисовать стрелку
    needle_angle = d_angle * math.pi / 2 + (math.pi * speed / MAX_SPEED * d_width)
    needle_x = int(center[0] + (radius - 20) * math.cos(needle_angle))
    needle_y = int(center[1] + (radius - 20) * math.sin(needle_angle))
    cv2.line(img, center, (needle_x, needle_y), (0, 0, 255), 3)

# Load the YOLOv11 model (change the path to your specific model path)
model = YOLO("./yolo11_custom2.pt")
print(settings)

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BRIGHTNESS, -3)
cap.set(cv2.CAP_PROP_CONTRAST, 6)
cap.set(cv2.CAP_PROP_HUE, 0)
cap.set(cv2.CAP_PROP_SATURATION, 15)
cap.set(cv2.CAP_PROP_GAMMA, 125)
cap.set(cv2.CAP_PROP_EXPOSURE, -5)
cap.set(cv2.CAP_PROP_SETTINGS, 1)

rightSpeed = 0
leftSpeed = 0

speeds = []

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()    
    if not ret:
        print("Failed to grab frame")
        break

    display_height, display_width, _ = frame.shape

    # Perform detection on the current frame
    result = model.predict(source=frame, verbose=False)[0]

    speed = 0.0
    confidence = 0.0

    boxes = result.boxes    
    for box in boxes:
        confidence = box.conf[0]

        if confidence >= MIN_CONFIDENCE:
            x1, y1, x2, y2 = box.xyxy[0]
            label = box.cls[0]  # Class ID (index)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Calculate the area of the bounding box
            object_width = x2 - x1

            speed = min(object_width / display_width / MAX_SCOPE, 1.0)    

    #if len(boxes) == 0 or confidence < MIN_CONFIDENCE:
    #    speed = 1.0
    print(f"speed: {speed}, confidence: {confidence}")

    # Определяем среднюю скорость
    average_speed = 0.0
    if len(speeds) > MAX_SPEEDS_LEN:
        speeds.pop(0)
    speeds.append(speed)
    sum_speed = 0
    for sp in speeds:
        sum_speed = sum_speed + sp
    average_speed = sum_speed / len(speeds)


    draw_speedometer(frame, round(display_width/4), average_speed)
    draw_speedometer(frame, round(display_width/4 + display_width/2), average_speed)

    # Display the frame with bounding boxes and labels
    cv2.imshow("Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# Release the capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
