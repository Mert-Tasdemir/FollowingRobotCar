import RPi.GPIO as GPIO
import time
from ultralytics import YOLO
import cv2

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Define motor control pins
LEFT_IN1 = 17  # Left motor direction pin 1
LEFT_IN2 = 27  # Left motor direction pin 2
RIGHT_IN3 = 22  # Right motor direction pin 1
RIGHT_IN4 = 23  # Right motor direction pin 2
EN_A = 18  # Left motor speed control (PWM)
EN_B = 19  # Right motor speed control (PWM)

# Set up motor pins as outputs
GPIO.setup([LEFT_IN1, LEFT_IN2, RIGHT_IN3, RIGHT_IN4, EN_A, EN_B], GPIO.OUT)

# Set up PWM for motor speed control
left_pwm = GPIO.PWM(EN_A, 100)  # 100 Hz frequency
right_pwm = GPIO.PWM(EN_B, 100)

# Start PWM with 0 duty cycle (motors off)
left_pwm.start(0)
right_pwm.start(0)

# Load YOLOv11 model
model = YOLO("D:\\yolo11_custom2\\yolo11_custom2.pt")

# Open webcam (or use Raspberry Pi Camera)
cap = cv2.VideoCapture(0)

# Frame dimensions and thresholds
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_centerX = frame_width / 2
frame_centerY = frame_height / 2
threshold = 50  
optimal_area = 20000
max_speed=100

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Perform detection on the frame
        results = model(frame)
        result = results[0]

        # Analyze bounding boxes
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width = x2 - x1
            height = y2 - y1
            area = width * height
            target_centerX = x1 + width / 2
            target_centerY = y1 + height / 2

            # Determine speeds based on area
            if area < optimal_area:  
                rightSpeed=leftSpeed=max_speed-(area/optimal_area)*max_speed
            else:
                leftSpeed = rightSpeed = 0  # Stop the car

            #left, right
            if target_centerX<frame_centerX-threshold:
                print("MOVE TO LEFT")
                if leftSpeed>0:
                    leftSpeed-=25
                else:
                    rightSpeed+=25    

            elif target_centerX>frame_centerX+threshold:
                print("MOVE TO RIGHT")
                if rightSpeed>0:
                    rightSpeed-=25
                else:
                    leftSpeed+=25
            else:
                print("CENTERED")

            # Ensure speeds are within valid range
            leftSpeed = max(0, min(leftSpeed, 100))
            rightSpeed = max(0, min(rightSpeed, 100))

            # Set motor directions and speeds
            if leftSpeed > 0:
                GPIO.output(LEFT_IN1, GPIO.HIGH)
                GPIO.output(LEFT_IN2, GPIO.LOW)
            else:
                GPIO.output(LEFT_IN1, GPIO.LOW)
                GPIO.output(LEFT_IN2, GPIO.LOW)

            if rightSpeed > 0:
                GPIO.output(RIGHT_IN3, GPIO.HIGH)
                GPIO.output(RIGHT_IN4, GPIO.LOW)
            else:
                GPIO.output(RIGHT_IN3, GPIO.LOW)
                GPIO.output(RIGHT_IN4, GPIO.LOW)

            # Apply speeds via PWM
            left_pwm.ChangeDutyCycle(leftSpeed)
            right_pwm.ChangeDutyCycle(rightSpeed)

        # Display the frame
        cv2.imshow("Detection", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Exiting...")

finally:
    # Cleanup
    left_pwm.stop()
    right_pwm.stop()
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()
