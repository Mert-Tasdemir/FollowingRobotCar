from ultralytics import YOLO
from ultralytics import settings
import cv2
import numpy as np
import math
from PIL import ImageFont, ImageDraw, Image

# Constants
MIN_CONFIDENCE = 0.54
MAX_SPEED = 100.0
STACK_LEN = 10

MAX_TARGET = 0.30
MIN_TARGET = 0.08
SCOPE_TARGET = 1.0 / (MAX_TARGET - MIN_TARGET)

DSLIP_X = 0.90
SCOPE_SLIP_X = 1.0 / DSLIP_X


def draw_target(target_size):
    x1, y1, x2, y2 = target_size
    if confidence < MIN_CONFIDENCE * 1.4:
        target_color = (0, 0, 255)
    else:
        target_color = (0, 255, 0)
    line_thickness = 2

    # Create an overlay for transparency
    overlay = frame.copy()

    # Draw rectangle
    corner = min(x2 - x1, y2 - y1) // 4
    cv2.rectangle(overlay, (x1, y1), (x2, y2), target_color, line_thickness // 2)
    cv2.line(frame, (x1, y1), (x1 + corner, y1), target_color, line_thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner), target_color, line_thickness)
    cv2.line(frame, (x2, y1), (x2 - corner, y1), target_color, line_thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner), target_color, line_thickness)
    cv2.line(frame, (x1, y2), (x1 + corner, y2), target_color, line_thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner), target_color, line_thickness)
    cv2.line(frame, (x2, y2), (x2 - corner, y2), target_color, line_thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner), target_color, line_thickness)


    # Draw crosshair lines
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    crosshair_length = min(x2 - x1, y2 - y1) // 4
    cv2.line(frame, (cx - crosshair_length, cy), (cx + crosshair_length, cy), target_color, line_thickness)
    cv2.line(frame, (cx, cy - crosshair_length), (cx, cy + crosshair_length), target_color, line_thickness)

    # Blend the overlay with the original frame
    alpha = 0.2
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_speedometer(img, x, average_speed):
    """
    Draws a speedometer on the image with a blurred dark background.

    Args:
        img: Input image.
        x: X-coordinate for the speedometer center.
        speed: Current speed (0-MAX_SPEED).
    """
    height, width, _ = img.shape
    line_width = 2
    radius = 100
    center = (x - line_width // 2, height - radius - line_width // 2 - 20)
    d_angle = -2.45
    d_width = 1.45

    speed = average_speed * MAX_SPEED

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
        grey = int(255*average_speed)
        red  = int(255*(1.0 - average_speed))
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

def calculate_speed(target_size, display_width):
    """Calculates the speed based on the bounding box size."""
    x1, _, x2, _ = target_size
    target_width = x2 - x1
    raw_speed = target_width / display_width
    speed = max(raw_speed - MIN_TARGET, 0.0)
    speed = min(speed * SCOPE_TARGET, 1.0)
    return 1.0 - speed * speed

def calculate_slip_x(target_size, display_width):
    x1, _, x2, _ = target_size
    display_center_x = display_width / 2
    target_slip_x = x1 + (x2 - x1) / 2
    raw_slip_x = (target_slip_x - display_center_x) / display_center_x
    if raw_slip_x > 0.0:
        slip_x = min(raw_slip_x * SCOPE_SLIP_X, 1.0)
        slip_x = slip_x * slip_x
    else:
        slip_x = max(raw_slip_x * SCOPE_SLIP_X, -1.0)
        slip_x = -1 * slip_x * slip_x
    return slip_x

def update_average_speed(speeds, speed):
    """Updates the rolling average speed."""
    if len(speeds) > STACK_LEN:
        speeds.pop(0)
    speeds.append(speed)
    return sum(speeds) / len(speeds)

def update_average_slip_x(slips, slip_x):
    """Updates the rolling average slip_x."""
    if len(slips) > STACK_LEN:
        slips.pop(0)
    slips.append(slip_x)
    return sum(slips) / len(slips)

# Load YOLO model
model = YOLO("./yolo11_custom2.pt")
print(settings)

# Initialize camera
cap = initialize_camera()

# Speed tracking variables
speeds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
slips = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    _, display_width, _ = frame.shape

    # Perform detection
    result = model.predict(source=frame, verbose=False)[0]
    boxes = result.boxes

    # Get highest confidence box
    index, confidence = get_highest_confidence_box(boxes)
    is_detected = index > -1 and confidence >= MIN_CONFIDENCE

    speed = 0.0
    slip_x = 0.0
    if is_detected:
        box = boxes[index]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        speed = calculate_speed((x1, y1, x2, y2), display_width)
        slip_x = calculate_slip_x((x1, y1, x2, y2), display_width)

        # Draw target overlay using a rectangular frame
        draw_target((x1, y1, x2, y2))

    if not is_detected:
        speed = 0.0

    # Update average speed
    average_speed = update_average_speed(speeds, speed)

    # Update average slip_x
    average_slip_x = update_average_slip_x(slips, slip_x)

    if average_slip_x > 0.0:
        speed_right = average_speed - average_slip_x
        speed_left = average_speed
    else:
        speed_right = average_speed
        speed_left = average_speed + average_slip_x

    # print(f"average_slip_x: {average_slip_x:.2f}, average_speed: {average_speed:.2f}, speed_left: {speed_left:.2f}, speed_right: {speed_right:.2f}")


    # Draw speedometers
    draw_speedometer(frame, round(display_width / 4), speed_left)
    draw_speedometer(frame, round(display_width / 4 + display_width / 2), speed_right)

    # Display the frame
    cv2.imshow("Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
