from ultralytics import YOLO
from ultralytics import settings
import cv2
import numpy as np
import math
import logging


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

    # Нарисовать круг
    cv2.circle(img, center, radius, (255, 255, 255), line_width)

    # Отметки на шкале (каждые 10 единиц)
    for i in range(0, 101, 10):
        angle = d_angle * math.pi / 2 + (math.pi * i / 100 * d_width)
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
    needle_angle = d_angle * math.pi / 2 + (math.pi * speed / 100 * d_width)
    needle_x = int(center[0] + (radius - 20) * math.cos(needle_angle))
    needle_y = int(center[1] + (radius - 20) * math.sin(needle_angle))
    cv2.line(img, center, (needle_x, needle_y), (0, 0, 255), 3)


# Установить уровень логирования
logging.getLogger().setLevel(logging.NOTSET)

# Load the YOLOv11 model (change the path to your specific model path)
model = YOLO("./yolo11_custom2.pt")
#results = model.predict(source='0', verbose=False, show=True)
#settings.update({"runs_dir": "/home/mert/sources/FollowingRobotCar"})

print(settings)


# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BRIGHTNESS, -3)
cap.set(cv2.CAP_PROP_CONTRAST, 20)
cap.set(cv2.CAP_PROP_HUE, 0)
cap.set(cv2.CAP_PROP_SATURATION, 15)
cap.set(cv2.CAP_PROP_GAMMA, 143)
cap.set(cv2.CAP_PROP_EXPOSURE, -5)
cap.set(cv2.CAP_PROP_SETTINGS, 1)

# Get the width and height of the frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_centerX=frame_width/2
frame_centerY=frame_height/2
left_thresholdX = frame_centerX - 50
right_thresholdX = frame_centerX + 50
print(f"Frame Size: {frame_width}x{frame_height}")
threshold=50
optimal_area = 20000
speed=0
leftSpeed=0
rightSpeed=0
max_speed=100

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()    
    if not ret:
        print("Failed to grab frame")
        break

    # Perform detection on the current frame
    results = model.predict(source=frame, verbose=False)

    #speed_v = 34

    # Отрисовать спидометр
    #draw_speedometer(frame, speed_v)

    # Access the first result (since it's returned as a list of results)
    result = results[0]

    #cv2.circle(frame, (int(frame_centerX), int(frame_centerY)), 5, (255, 0, 0), -1)  # Absolute center (blue)
    #cv2.putText(frame, "Center", (int(frame_centerX) + 10, int(frame_centerY) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    #cv2.circle(frame, (int(left_thresholdX), int(frame_centerY)), 5, (0, 255, 0), -1)  # Left threshold (green)
    #cv2.putText(frame, "Left Threshold", (int(left_thresholdX) + 10, int(frame_centerY) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #cv2.circle(frame, (int(right_thresholdX), int(frame_centerY)), 5, (0, 0, 255), -1)  # Right threshold (red)
    #cv2.putText(frame, "Right Threshold", (int(right_thresholdX) + 10, int(frame_centerY) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Get the bounding boxes and labels
    
    boxes = result.boxes  # Detected bounding boxes
    for box in boxes:
        # Get the bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        label = box.cls[0]  # Class ID (index)
        confidence = box.conf[0]  # Confidence score

        # Calculate the area of the bounding box
        width = x2 - x1
        height = y2 - y1
        area = width * height

        # Draw the bounding box on the frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Add the label (class name, confidence, and area)
        #cv2.putText(frame, f"Target {confidence:.2f} Area: {int(area)}",
        #            (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
        #            (0, 255, 0), 2)

        # Print the area of the target in the console
        #print(f"Detected Target - Area: {int(area)}, Confidence: {confidence:.2f}")
        target_centerX=x1+(x2-x1)/2
        target_centerY=y1+(y2-y1)/2
        #print(f"x1={x1},y1={y1} and x2={x2},y2={y2}")
        #print(f"center of target={target_centerX},{target_centerY}")
        #print(f"center of frame={frame_centerX},{frame_centerY}")
        #print(f"frame_width={frame_width} and frame_height={frame_height}")

        #cv2.circle(frame, (int(target_centerX), int(target_centerY)), 5, (255, 255, 255), -1)
        #cv2.putText(frame, "Target Center", (int(target_centerX) + 10, int(target_centerY) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        rightSpeed = 10 
           
        #speed
        if area < optimal_area:  
            speed=max_speed-(area/optimal_area)*max_speed
            leftSpeed=rightSpeed=int(speed)
        else:
            #print(f"area({area}) > 20000")
            #print("NO FORWARD")
            leftSpeed=0
            #rightSpeed=0

        #left, right
        if target_centerX<frame_centerX-threshold:
            #print("MOVE TO LEFT")
            if leftSpeed>0:
                leftSpeed-=25
            else:
                rightSpeed+=25    

        elif target_centerX>frame_centerX+threshold:
            #print("MOVE TO RIGHT")
            if rightSpeed>0:
                rightSpeed-=25
            else:
                leftSpeed+=25
#        else:
#            print("CENTERED")

        #print(f"leftSpeed={leftSpeed} and rightSpeed={rightSpeed}")
        
        # Display leftSpeed and rightSpeed on the frame
        # cv2.putText(frame, f"Left Speed: {leftSpeed}", (10, frame_height - 40), 
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # cv2.putText(frame, f"Right Speed: {rightSpeed}", (10, frame_height - 10), 
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
  

    height, width, _ = frame.shape
    draw_speedometer(frame, round(width/4), rightSpeed)
    draw_speedometer(frame, round(width/4 + width/2), rightSpeed)

    # Display the frame with bounding boxes and labels
    cv2.imshow("Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# Release the capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
