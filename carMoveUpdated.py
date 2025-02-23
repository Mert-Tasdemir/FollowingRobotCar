from ultralytics import YOLO
from ultralytics import settings
import cv2
import numpy as np
import RPi.GPIO as GPIO

# Define motor control pins (Raspberry Pi GPIO pins)
LEFT_IN1 = 23  # Left motor direction pin 1 (connect to IN1 on L298N)
LEFT_IN2 = 24  # Left motor direction pin 2 (connect to IN2 on L298N)
EN_A = 25      # Left motor speed control (PWM) (connect to ENA on L298N)

RIGHT_IN3 = 22  # Right motor direction pin 1 (connect to IN3 on L298N)
RIGHT_IN4 = 27  # Right motor direction pin 2 (connect to IN4 on L298N)
EN_B = 17       # Right motor speed control (PWM) (connect to ENB on L298N)

# Initialize GPIO
GPIO.setmode(GPIO.BCM)  # Use Broadcom pin numbering

# Set motor control pins as outputs
GPIO.setup(LEFT_IN1, GPIO.OUT)
GPIO.setup(LEFT_IN2, GPIO.OUT)
GPIO.setup(EN_A, GPIO.OUT)

GPIO.setup(RIGHT_IN3, GPIO.OUT)
GPIO.setup(RIGHT_IN4, GPIO.OUT)
GPIO.setup(EN_B, GPIO.OUT)

# Set up PWM for speed control
LEFT_PWM = GPIO.PWM(EN_A, 100)  # 100 Hz PWM frequency
RIGHT_PWM = GPIO.PWM(EN_B, 100)

LEFT_PWM.start(0)  # Start with 0% duty cycle
RIGHT_PWM.start(0)

# Load the YOLOv11 model (change the path to your specific model path)
model = YOLO("./yolo11_custom2.pt")
settings.update({"runs_dir": "/home/mert/sources/FollowingRobotCar"})

print(settings)

# Open webcam
cap = cv2.VideoCapture(0)  # '0' for the default webcam

# Get the width and height of the frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_centerX=frame_width/2
frame_centerY=frame_height/2
left_thresholdX = frame_centerX - 50
right_thresholdX = frame_centerX + 50
print(f"Frame Size: {frame_width}x{frame_height}")
THRESHOLD=50
speed=0
OPTIMAL_WIDTH=250
leftSpeed=0
rightSpeed=0
MIN_CONFIDENCE=0.5
MAX_SPEED=100
MIN_SPEED=0
MAX_DIRECTION_SPEED=25
MIN_WIDTH=50

left_limitX = 125
right_limitX = frame_width - 125

left_speeds = [0] * 10
right_speeds = [0] * 10
i=0
calculated_leftSpeed=0
calculated_rightSpeed=0

class Motors:
    
    def set_motor_speed(calculated_leftSpeed, calculated_rightSpeed):
        
        calculated_leftSpeed=max(0, min(100, calculated_leftSpeed))
        calculated_rightSpeed=max(0, min(100, calculated_rightSpeed))
        
        #left motor
        if calculated_leftSpeed>0:
            GPIO.output(LEFT_IN1, GPIO.HIGH)
            GPIO.output(LEFT_IN2, GPIO.LOW)
        else:
            GPIO.output(LEFT_IN1, GPIO.LOW)
            GPIO.output(LEFT_IN2, GPIO.LOW)
        #right motor
        if calculated_rightSpeed>0:
            GPIO.output(RIGHT_IN3, GPIO.HIGH)
            GPIO.output(RIGHT_IN4, GPIO.LOW)
        else:
            GPIO.output(RIGHT_IN3, GPIO.LOW)
            GPIO.output(RIGHT_IN4, GPIO.LOW)
        
        
        #GPIO.output(LEFT_IN1, GPIO.HIGH if calculated_leftSpeed > 0 else GPIO.LOW)
        #GPIO.output(LEFT_IN2, GPIO.LOW)

        #GPIO.output(RIGHT_IN3, GPIO.HIGH if calculated_rightSpeed > 0 else GPIO.LOW)
        #GPIO.output(RIGHT_IN4, GPIO.LOW)
        
        LEFT_PWM.ChangeDutyCycle(calculated_leftSpeed)
        RIGHT_PWM.ChangeDutyCycle(calculated_rightSpeed)        

class Calculate:
    
    def get_highest_confidence(boxes):
        if len(boxes) == 0:
            return -1, None  # No objects detected

        # Find the index of the highest confidence box
        highest_index = max(range(len(boxes)), key=lambda i: boxes[i].conf[0])
        highest_confidence = float(boxes[highest_index].conf[0])  # Convert to float

        return highest_index, highest_confidence

    def adjust_direction_speed(target_centerX, thresholdX, edge):
                if target_centerX<left_limitX or target_centerX>right_limitX:
                    return MAX_DIRECTION_SPEED
                else:
                    return MAX_DIRECTION_SPEED * (target_centerX - thresholdX) / (edge - thresholdX)                

    #def apply_direction_speed(subtract_from_this, add_to_this, directionSpeed):
        #if subtract_from_this>=directionSpeed:
         #   subtract_from_this=(int)(subtract_from_this-directionSpeed)
        #elif subtract_from_this<directionSpeed:
         #   directionSpeed=directionSpeed-subtract_from_this
          #  subtract_from_this=0
           # add_to_this=(int)(add_to_this+directionSpeed)



     #   return subtract_from_this, add_to_this

    def apply_direction_speed(subtract_from_this, directionSpeed): 
        return max(0, int(subtract_from_this-directionSpeed))

class Draw:
    
    def draw_speedometer(frame, x, y, speed, label="Left"):
        
        # Create a blurred version of the frame
        blurred_frame = cv2.GaussianBlur(frame, (101, 101), 50)  

        # Create a mask for the speedometer circle
        mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.circle(mask, (x, y), 60, (255, 255, 255), -1)  # Filled circle (white on mask)

        # Blend the blurred and original frame using the mask
        speedometer_area = np.where(mask == 255, blurred_frame, frame)

        # Draw the blended speedometer area onto the original frame
        frame[:, :] = speedometer_area

        # Draw the speedometer outline
        cv2.circle(frame, (x, y), 50, (255, 255, 255), 2)

        start_angle = 0
        end_angle = speed * 360 / 100
        color=255-speed/100*255
        cv2.ellipse(frame, (x, y), (50, 50), 0, start_angle, end_angle, (color, color, 255), 5)
        cv2.putText(frame, f"{int(speed)}", (x - 15, y + 10), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, label, (x - 25, y + 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_bounding_box(frame, x1, y1, x2, y2, line_length, color, confidence, width, 
                        target_centerX, target_centerY, left_thresholdX, right_thresholdX, frame_centerY):
        cv2.line(frame,(x1,y1),(x1+line_length,y1), color, 2)
        cv2.line(frame,(x1,y1),(x1,y1+line_length), color, 2)
            
        cv2.line(frame,(x2,y2),(x2, y2-line_length), color, 2)
        cv2.line(frame,(x2,y2),(x2-line_length,y2), color, 2)

        cv2.line(frame,(x2,y1),(x2-line_length,y1), color, 2)
        cv2.line(frame,(x2,y1),(x2,y1+line_length), color, 2)
            
        cv2.line(frame,(x1,y2),(x1,y2-line_length), color, 2)
        cv2.line(frame,(x1,y2),(x1+line_length,y2), color, 2)

        cv2.line(frame,(target_centerX-line_length//2,target_centerY),(target_centerX+line_length//2,target_centerY), color, 2)
        cv2.line(frame, (target_centerX,target_centerY-line_length//2), (target_centerX, target_centerY+line_length//2,), color, 2)

        if target_centerX<left_thresholdX:
            cv2.line(frame, (int(left_thresholdX), int(frame_centerY)), (int(target_centerX),int(target_centerY)), color, 2)
        elif target_centerX>right_thresholdX:
            cv2.line(frame, (int(right_thresholdX), int(frame_centerY)), (int(target_centerX),int(target_centerY)), color, 2)    
        #cv2.circle(frame, (int(target_centerX), int(target_centerY)), 5, (255, 255, 255), -1)
        cv2.putText(frame, f"Target {confidence:.2f} width: {int(width)}",
                    (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    color, 2)

    def draw_center_circles(frame, frame_centerX, frame_centerY):
        cv2.circle(frame, (int(frame_centerX), int(frame_centerY)), 5, (0, 255, 0), -1)
        cv2.circle(frame, (int(left_thresholdX), int(frame_centerY)), 5, (0, 255, 220), -1)
        cv2.circle(frame, (int(right_thresholdX), int(frame_centerY)), 5, (0, 255, 220), -1)

try:
    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break

        # Perform detection on the current frame
        results = model(frame)

        # Access the first result (since it's returned as a list of results)
        result = results[0]

        #center circles
        Draw.draw_center_circles(frame, frame_centerX, frame_centerY)

        # Get the bounding boxes and labels
        boxes = result.boxes
        index, confidence = Calculate.get_highest_confidence(boxes)
        if index!=-1 and confidence>=MIN_CONFIDENCE:
            box = boxes[index]
            
            # Get the bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = box.cls[0]  # Class ID (index)

            width = x2 - x1

            #bounding box color based on confidence level
            green_value = 255 * ((confidence - 0.5) / (1 - 0.5))
            red_value = 255 - green_value
            color = (0, green_value, red_value)
        
            target_centerX=int(x1+(x2-x1)/2)
            target_centerY=int(y1+(y2-y1)/2)

            #cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            line_length = (x2-x1)//5
    
            Draw.draw_bounding_box(frame, x1, y1, x2, y2, line_length, color, confidence, width, 
                        target_centerX, target_centerY, left_thresholdX, right_thresholdX, frame_centerY)

            directionSpeed=0

            #speed
            if width > OPTIMAL_WIDTH:
                speed=0
            elif width < MIN_WIDTH:
                speed = MAX_SPEED
            else:
                speed = ((OPTIMAL_WIDTH - width) / (OPTIMAL_WIDTH - MIN_WIDTH)) * MAX_SPEED
                speed = int(speed)
            leftSpeed=rightSpeed=speed   

            #left, right
            if target_centerX < left_thresholdX:
                directionSpeed = Calculate.adjust_direction_speed(target_centerX, left_thresholdX, left_limitX)
                #leftSpeed, rightSpeed = Calculate.apply_direction_speed(leftSpeed, rightSpeed, directionSpeed)
                leftSpeed = Calculate.apply_direction_speed(leftSpeed, directionSpeed)

            elif target_centerX > right_thresholdX:
                directionSpeed = Calculate.adjust_direction_speed(target_centerX, right_thresholdX, right_limitX)
                #rightSpeed, leftSpeed = Calculate.apply_direction_speed(rightSpeed, leftSpeed, directionSpeed)
                rightSpeed = Calculate.apply_direction_speed(rightSpeed, directionSpeed)       
            
        else:
            leftSpeed=rightSpeed=0

        left_speeds[i]=leftSpeed
        right_speeds[i]=rightSpeed
        i=(i+1)%len(left_speeds)
        
        calculated_leftSpeed = sum(left_speeds) / len(left_speeds)
        calculated_rightSpeed = sum(right_speeds) / len(right_speeds)

        Motors.set_motor_speed(calculated_leftSpeed, calculated_rightSpeed)

        Draw.draw_speedometer(frame, 55, frame_height - 100, calculated_leftSpeed, label="Left")
        Draw.draw_speedometer(frame, frame_width-55, frame_height - 100, calculated_rightSpeed, label="Right")    

        # Convert speed lists to string format
        left_speeds_text = "Left: " + ", ".join(map(str, left_speeds))
        right_speeds_text = "Right: " + ", ".join(map(str, right_speeds))
        # Display the lists on the frame
        cv2.putText(frame, left_speeds_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, right_speeds_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
          
        # Display the frame with bounding boxes and labels
        cv2.imshow("Detection", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    print("Cleaning up GPIO...")
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()
