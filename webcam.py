from ultralytics import YOLO
from ultralytics import settings
import cv2
import numpy as np
import math
from PIL import ImageFont, ImageDraw, Image

# Constants
MIN_CONFIDENCE = 0.54
MAX_SPEED = 100.0
MAX_SPEEDS_LEN = 10

MAX_OBJECT = 0.30
MIN_OBJECT = 0.08
SCOPE_OBJECT = 1.0 / (MAX_OBJECT - MIN_OBJECT)

def draw_speedometer(img, x, speed):
    """
    Draws a speedometer on the image with a blurred dark background.

    Args:
        img: Input image.
        x: X-coordinate for the speedometer center.
        speed: Current speed (0-100).
    """
    height, width, _ = img.shape
    line_width = 2
    radius = 100
    center = (x - line_width // 2, height - radius - line_width // 2 - 20)
    d_angle = -2.45
    d_width = 1.45

    speed = (1.0 - speed) * MAX_SPEED

    # Create a mask for the blurred background circle
    mask = np.zeros_like(img, dtype=np.uint8)
    bg_radius = radius + 20
    cv2.circle(mask, center, bg_radius, (255, 255, 255), -1)

    # Apply the blur and darken only inside the mask
    overlay = img.copy()
    cv2.circle(overlay, center, bg_radius, (0, 0, 0), -1)
    alpha = 0.2  # Transparency factor for darkening
    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    blurred_overlay = cv2.GaussianBlur(blended, (7, 7), 0)
    img[mask[:, :, 0] == 255] = blurred_overlay[mask[:, :, 0] == 255]

    # Draw circle
    cv2.circle(img, center, radius, (255, 255, 255), line_width)

    # Draw scale markings
    for i in range(0, 101, 10):
        angle = d_angle * math.pi / 2 + (math.pi * i / MAX_SPEED * d_width)
        x1 = int(center[0] + radius * math.cos(angle))
        y1 = int(center[1] + radius * math.sin(angle))
        x2 = int(center[0] + (radius - 10) * math.cos(angle))
        y2 = int(center[1] + (radius - 10) * math.sin(angle))
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), line_width)

        # Add labels
        label_x = int(center[0] + (radius - 25) * math.cos(angle))
        label_y = int(center[1] + (radius - 25) * math.sin(angle))
        cv2.putText(img, str(i), (label_x - 10, label_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (255, 255, 255), 1)

    # Draw needle
    needle_angle = d_angle * math.pi / 2 + (math.pi * speed / MAX_SPEED * d_width)
    needle_x = int(center[0] + (radius - 20) * math.cos(needle_angle))
    needle_y = int(center[1] + (radius - 20) * math.sin(needle_angle))
    cv2.line(img, center, (needle_x, needle_y), (0, 0, 255), 3)

    # Draw speed value
    if speed > 0.0:
        grey = int(255*(speed/MAX_SPEED))
        red  = int(255*(1-speed/MAX_SPEED))
        color = (grey, grey, red+grey)
        speed_value = str(int(speed))
        text_size = cv2.getTextSize(speed_value, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = center[0] - text_size[0] // 2
        text_y = center[1] + radius // 2 + text_size[1]
        cv2.putText(img, speed_value, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)    

def initialize_camera():
    """Initializes and configures the webcam."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, -3)
    cap.set(cv2.CAP_PROP_CONTRAST, 6)
    cap.set(cv2.CAP_PROP_HUE, 0)
    cap.set(cv2.CAP_PROP_SATURATION, 15)
    cap.set(cv2.CAP_PROP_GAMMA, 125)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    cap.set(cv2.CAP_PROP_SETTINGS, 1)
    return cap

def get_highest_confidence_box(boxes):
    """Finds the box with the highest confidence."""
    highest_confidence = 0.0
    index = -1
    for i, box in enumerate(boxes):
        conf = box.conf[0]
        if conf > highest_confidence:
            highest_confidence = conf
            index = i
    return index, highest_confidence

def calculate_speed(box, display_width):
    """Calculates the speed based on the bounding box size."""
    x1, _, x2, _ = box.xyxy[0]
    object_width = x2 - x1
    raw_speed = object_width / display_width
    speed = max(raw_speed - MIN_OBJECT, 0.0)
    speed = min(speed * SCOPE_OBJECT, 1.0)
    return speed * speed

def update_average_speed(speeds, speed):
    """Updates the rolling average speed."""
    if len(speeds) > MAX_SPEEDS_LEN:
        speeds.pop(0)
    speeds.append(speed)
    return sum(speeds) / len(speeds)

# Load YOLO model
model = YOLO("./yolo11_custom2.pt")
print(settings)

# Initialize camera
cap = initialize_camera()

# Speed tracking variables
speeds = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    display_height, display_width, _ = frame.shape

    # Perform detection
    result = model.predict(source=frame, verbose=False)[0]
    boxes = result.boxes

    # Get highest confidence box
    index, confidence = get_highest_confidence_box(boxes)
    is_detected = index > -1 and confidence >= MIN_CONFIDENCE

    speed = 0.0
    if is_detected:
        box = boxes[index]
        speed = calculate_speed(box, display_width)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if not is_detected:
        speed = 1.0

    # Update average speed
    average_speed = update_average_speed(speeds, speed)

    # Draw speedometers
    draw_speedometer(frame, round(display_width / 4), average_speed)
    draw_speedometer(frame, round(display_width / 4 + display_width / 2), average_speed)

    # Display the frame
    cv2.imshow("Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
