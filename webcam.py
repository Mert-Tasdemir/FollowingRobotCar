from ultralytics import YOLO
from ultralytics import settings
import cv2


import draws


# Constants
MIN_CONFIDENCE = 0.54
MAX_SPEED = 100.0
STACK_LEN = 10

MAX_TARGET = 0.30
MIN_TARGET = 0.08
SCOPE_TARGET = 1.0 / (MAX_TARGET - MIN_TARGET)

DSLIP_X = 0.90
SCOPE_SLIP_X = 1.0 / DSLIP_X

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
        draws.draw_target(frame, (x1, y1, x2, y2), confidence, MIN_CONFIDENCE)

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
    draws.draw_speedometer(frame, round(display_width / 4), speed_left, MAX_SPEED)
    draws.draw_speedometer(frame, round(display_width / 4 + display_width / 2), speed_right, MAX_SPEED)

    # Display the frame
    cv2.imshow("Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
