from ultralytics import YOLO
from ultralytics import settings
import cv2
import libs.draws as draws
import libs.calculator as calculator
from picamera2 import Picamera2

# Constants
MIN_CONFIDENCE = 0.54

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480


def initialize_camera2():
    """Initializes and configures the webcam."""
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (DISPLAY_WIDTH, DISPLAY_HEIGHT)
    picam2.preview_configuration.main.format = "RGB888"
    # picam2.preview_configuration.align()
    # picam2.configure("preview")
    return picam2


cap = initialize_camera2()
cap.start()

# Load YOLO model
# model = YOLO("./yolo11n.pt")
# print(settings)

# Export the model to NCNN format
# model.export(format="ncnn")  # creates 'yolo11n_ncnn_model'

# Load the exported NCNN model
model = YOLO("yolo11n_ncnn_model", task='detect')


# Speed tracking variables
speeds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
slips = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

while True:
    frame = cap.capture_array()

    # Run YOLO11 inference on the frame
    # results = model(frame)
    results = model.predict(source=frame, verbose=False, conf=MIN_CONFIDENCE)

    # Visualize the results on the frame
    result = results[0]
    # frame = result.plot()

    _, display_width, _ = frame.shape

    # Perform detection
    # result = model.predict(source=frame, verbose=False)[0]
    boxes = result.boxes

    # Get highest confidence box
    index, confidence = calculator.get_highest_confidence_box(boxes)
    is_detected = index > -1 and confidence >= MIN_CONFIDENCE

    speed = 0.0
    slip_x = 0.0
    if is_detected:
        box = boxes[index]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        speed = calculator.calculate_speed((x1, y1, x2, y2), display_width)
        slip_x = calculator.calculate_slip_x((x1, y1, x2, y2), display_width)

        # Draw target overlay using a rectangular frame
        if confidence < MIN_CONFIDENCE * 1.4:
            target_color = (0, 0, 255)
        else:
            target_color = (0, 255, 0)
        draws.draw_target(frame, (x1, y1, x2, y2), target_color)

    if not is_detected:
        speed = 0.0

    # Update average speed
    average_speed = calculator.update_average_speed(speeds, speed)

    # Update average slip_x
    average_slip_x = calculator.update_average_slip_x(slips, slip_x)

    if average_slip_x > 0.0:
        speed_right = max(average_speed - average_slip_x, 0)
        speed_left = average_speed
    else:
        speed_right = average_speed
        speed_left = max(average_speed + average_slip_x, 0)
    # print(f"average_slip_x: {average_slip_x:.2f}, average_speed: {average_speed:.2f}, speed_left: {speed_left:.2f}, speed_right: {speed_right:.2f}")

    # Draw speedometers
    draws.draw_speedometer(frame, round(display_width / 4), speed_left)
    draws.draw_speedometer(frame, round(
        display_width / 4 + display_width / 2), speed_right)

    # Display the frame
    cv2.imshow("Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
