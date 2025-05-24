from ultralytics import YOLO
from ultralytics import settings
import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
import os
from flask import Flask, Response, render_template, jsonify
from flask_socketio import SocketIO, emit
import threading
import time

app = Flask(__name__, template_folder='/home/mert/source/FollowingRobotCar/templates')
socketio = SocketIO(app)

pursuit_active = False
manual_mode =False

# Shared frame for streaming
processed_frame = None
frame_lock = threading.Lock()

# Define motor control pins (Raspberry Pi GPIO pins)
LEFT_IN1 = 24  #left forward
LEFT_IN2 = 23  #left backward
EN_A = 25      #left PWM
             
RIGHT_IN3 = 22 #right forward
RIGHT_IN4 = 27 #right backward
EN_B = 17      #right PWM

# Initialize GPIO
GPIO.setmode(GPIO.BCM)  # Use Broadcom pin numbering

MOTOR_PINS = [LEFT_IN1, LEFT_IN2, EN_A, RIGHT_IN3, RIGHT_IN4, EN_B]

for pin in MOTOR_PINS:
    GPIO.setup(pin, GPIO.OUT)


FREQUENCY = 40
# Set up PWM for speed control
LEFT_PWM = GPIO.PWM(EN_A, FREQUENCY)  # FREQUENCY Hz PWM frequency
RIGHT_PWM = GPIO.PWM(EN_B, FREQUENCY)

LEFT_PWM.start(0)  # Start with 0% duty cycle
RIGHT_PWM.start(0)

# Load the YOLOv11 model (change the path to your specific model path)
#model = YOLO("./yolo11_custom2.pt")
#settings.update({"runs_dir": "/home/mert/source/FollowingRobotCar"})

# Load the exported NCNN model
model = YOLO(os.getcwd() + "/yolo11_ncnn_model", task='detect')

results = model.predict(source=None, verbose=False, conf=0) #perform detection on Nothing, to Load the model fully. without this line, when start is clicked in web, it takes long time at 
#Loading /home/mert/source/FollowingRobotCar/yolo11_ncnn_model for NCNN inference    line  (gives warning because of: source=None)

#print(settings)

# Open webcam
#cap = cv2.VideoCapture(0)  # '0' for the default webcam

# Get the width and height of the frame
frame_width = 640 #int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = 480 #int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def initialize_camera2():
    #Initializes and configures the webcam.
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (frame_width, frame_height)
    picam2.preview_configuration.main.format = "RGB888"
    # picam2.preview_configuration.align()
    # picam2.configure("preview")
    return picam2

cap = initialize_camera2()
cap.start()

"""
def initialize_camera2():
    #Initializes and configures the Raspberry Pi camera.
    picam2 = Picamera2()
    
    # Get default camera resolution
    camera_info = picam2.sensor_modes[0]  # First available mode
    frame_width, frame_height = camera_info["size"]

    # Configure camera
    picam2.preview_configuration.main.size = (frame_width, frame_height)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")

    return picam2, frame_width, frame_height

cap, frame_width, frame_height = initialize_camera2()
cap.start()
"""


debug_counter = 0

frame_centerX=frame_width/2
frame_centerY=frame_height/2
THRESHOLD=50
left_thresholdX = frame_centerX - THRESHOLD
right_thresholdX = frame_centerX + THRESHOLD
print(f"Frame Size: {frame_width}x{frame_height}")
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
list_len = 3
left_speeds = [0] * list_len
right_speeds = [0] * list_len
counter=0
calculated_leftSpeed=0
calculated_rightSpeed=0
no_detection_time = 5 #seconds
last_seen_time = time.time()
current_speed_factor_left = 1
current_speed_factor_right = 1
turn_coefficient = 1

owner_lost=False
detected=True #following


case_following=False
case_noDetection=False
case_targetLost=False



#generate_frames() starts on page load (when the <img src="/video"> is rendered)
#generate_frames() runs in its own thread, but that thread is created by Flask,
def generate_frames():
    global processed_frame
    while True:
        with frame_lock:
            if processed_frame is None:
                continue
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start():
    global pursuit_active, owner_lost, detected
    pursuit_active = True
    owner_lost = False
    detected = False
    emit_status()
    return "Started following"

@app.route('/stop')
def stop():
    global pursuit_active
    pursuit_active = False
    emit_status()
    return "Stopped following"

@app.route('/shutdown')
def shutdown():
    GPIO.cleanup()
    os.system("sudo shutdown now")
    return "Shutting down"

@app.route('/video')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# WebSocket event for when the client connects
@socketio.on('connect')
def handle_connect():
    emit_status()  # Emit the current status when the client connects

@socketio.on('set_speed')
def handle_set_speed(data):
    side = data.get('side')  # 'left' or 'right'
    value = float(data.get('value', 1.0))
    if side == 'left':
        global current_speed_factor_left
        current_speed_factor_left = float(data['value'])  #in range 0.0–1.0
    elif side == 'right':
        global current_speed_factor_right
        current_speed_factor_right = float(data['value'])  #in range 0.0–1.0
    

@socketio.on('set_list_len')
def handle_set_list_len(data):
    global list_len
    list_len = int(data['value'])
    initialize_speedLists()

@socketio.on('set_manual_mode')
def enter_manual_mode(data):
    global manual_mode
    manual_mode = bool(data['value'])
    left_speed = 0
    right_speed = 0
    if manual_mode:
        left_speed = int(data.get('leftSpeed', 0))
        right_speed = int(data.get('rightSpeed', 0))
    Motors.set_motor_speed(left_speed, right_speed)
    print(f"from set_manual_mode  left_speed={left_speed} and right_speed={right_speed}")


    




def initialize_speedLists():
    global left_speeds, right_speeds, counter
    left_speeds = [0] * list_len
    right_speeds = [0] * list_len
    counter=0

# This function emits the current system status
def emit_status():
    global pursuit_active, owner_lost, detected
    """Emit the status update to all connected clients."""
    #print("Emitting status:", {"pursuit_active": pursuit_active, "owner_lost": owner_lost, "detected": detected})
    if not pursuit_active:
        socketio.emit('status_update', {"message": "System Offline"})
    elif owner_lost:
        socketio.emit('status_update', {"message": "Target Lost"})
    elif detected:
        socketio.emit('status_update', {"message": "Following"})
    else:
        socketio.emit('status_update', {"message": "No Detection"})


def start_flask():
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)


# Start the Flask app in a separate thread
flask_thread = threading.Thread(target=start_flask, daemon=True)
flask_thread.start()




class Motors:
    Processing = False
    
    def set_motor_speed(calculated_leftSpeed, calculated_rightSpeed):
        if not Motors.Processing:
            Motors.Processing = True

            print(">>>>>>> MOTOR SPEED STARTED <<<<<<<")

            calculated_leftSpeed=max(0, min(100, calculated_leftSpeed))
            calculated_rightSpeed=max(0, min(100, calculated_rightSpeed))

            LEFT_PWM.ChangeDutyCycle(calculated_leftSpeed)
            RIGHT_PWM.ChangeDutyCycle(calculated_rightSpeed)        


            GPIO.output(LEFT_IN1, GPIO.HIGH)
            GPIO.output(LEFT_IN2, GPIO.LOW)
            
            GPIO.output(RIGHT_IN3, GPIO.HIGH)
            GPIO.output(RIGHT_IN4, GPIO.LOW)

            print(">>>>>>> MOTOR SPEED COMPLETED <<<<<<<")
        
    def clean():
        if Motors.Processing:
            Motors.Processing = False
            LEFT_PWM.ChangeDutyCycle(0)
            RIGHT_PWM.ChangeDutyCycle(0)
            GPIO.output(LEFT_IN1, GPIO.LOW)
            GPIO.output(LEFT_IN2, GPIO.LOW)
            GPIO.output(RIGHT_IN3, GPIO.LOW)
            GPIO.output(RIGHT_IN4, GPIO.LOW)
            

class Calculate:
    
    def get_highest_confidence(boxes):
        if len(boxes) == 0:
            return -1, None  # No objects detected

        # Find the index of the highest confidence box
        highest_index = max(range(len(boxes)), key=lambda i: boxes[i].conf[0])
        highest_confidence = float(boxes[highest_index].conf[0])  # Convert to float

        return highest_index, highest_confidence

    def calculate_turn_coefficient(width):
        return ((width-0)/(320-0))*(1-0)+0

    def adjust_direction_speed(target_centerX, thresholdX, edge, width):
                global turn_coefficient
                turn_coefficient = Calculate.calculate_turn_coefficient(width)
                if target_centerX<left_limitX or target_centerX>right_limitX:
                    return int(MAX_DIRECTION_SPEED * turn_coefficient)
                else:
                    return int(MAX_DIRECTION_SPEED * turn_coefficient * (target_centerX - thresholdX) / (edge - thresholdX))                
                
    def apply_direction_speed(subtract_from_this, directionSpeed): 
        return max(0, int(subtract_from_this-directionSpeed))

class Draw:
    
    def draw_speedometer(frame, x, y, speed, label="Left"):
        
        # Create a blurred version of the frame
        #blurred_frame = cv2.GaussianBlur(frame, (101, 101), 50)  

        # Create a mask for the speedometer circle
        #mask = np.zeros_like(frame, dtype=np.uint8)
        #cv2.circle(mask, (x, y), 60, (255, 255, 255), -1)  # Filled circle (white on mask)

        # Blend the blurred and original frame using the mask
        #speedometer_area = np.where(mask == 255, blurred_frame, frame)

        # Draw the blended speedometer area onto the original frame
        #frame[:, :] = speedometer_area

        # Draw the speedometer outline
        cv2.circle(frame, (x, y), 50, (255, 255, 255), 2)

        start_angle = 0
        end_angle = speed * 360 / 100
        color=255-speed/100*255
        cv2.circle(frame, (x, y), 55, (0,0,0), -1)
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

    #Motors.set_motor_speed(40, 40)
    # while True:
    #     print()   

    while True:
        
        if pursuit_active and not manual_mode:
            # Capture a frame from the webcam
            #ret, frame = cap.read()
            
            frame = cap.capture_array()

            #if not ret:
            #    print("Failed to grab frame")
            #    break

            # Perform detection on the current frame
            #results = model(frame)
            results = model.predict(source=frame, verbose=False, conf=MIN_CONFIDENCE) ## LOADING LINE ## but added it to the start to prevent long wait in loop (here)

            #debug_counter+=1
            #print(f"debug_counter:{debug_counter}")

            # Access the first result (since it's returned as a list of results)
            result = results[0]

            #center circles
            Draw.draw_center_circles(frame, frame_centerX, frame_centerY)

            # Get the bounding boxes and labels
            boxes = result.boxes
            index, confidence = Calculate.get_highest_confidence(boxes)
            if index!=-1 and confidence>=MIN_CONFIDENCE: #if there is a True Detection

                owner_lost=False
                detected=True
                if not case_following:
                    emit_status()
                    case_following=True
                    case_noDetection=False
                    case_targetLost=False
                    

                # Target is detected, reset the timer
                last_seen_time = time.time()

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
                    directionSpeed = Calculate.adjust_direction_speed(target_centerX, left_thresholdX, left_limitX, width)
                    #leftSpeed, rightSpeed = Calculate.apply_direction_speed(leftSpeed, rightSpeed, directionSpeed)
                    leftSpeed = Calculate.apply_direction_speed(leftSpeed, directionSpeed)

                elif target_centerX > right_thresholdX:
                    directionSpeed = Calculate.adjust_direction_speed(target_centerX, right_thresholdX, right_limitX, width)
                    #rightSpeed, leftSpeed = Calculate.apply_direction_speed(rightSpeed, leftSpeed, directionSpeed)
                    rightSpeed = Calculate.apply_direction_speed(rightSpeed, directionSpeed)       
                
            else:
                leftSpeed=rightSpeed=0
                if time.time() - last_seen_time < no_detection_time:
                    owner_lost=False
                    detected=False
                    if not case_noDetection:
                        emit_status()
                        case_following=False
                        case_noDetection=True
                        case_targetLost=False
                else:
                    owner_lost=True
                    detected=False
                    if not case_targetLost:
                        emit_status()
                        case_following=False
                        case_noDetection=False
                        case_targetLost=True
                        
            leftSpeed = int(leftSpeed * current_speed_factor_left)
            rightSpeed = int(rightSpeed * current_speed_factor_right)

            left_speeds[counter]=leftSpeed
            right_speeds[counter]=rightSpeed
            counter=(counter+1)%list_len #len(left_speeds)
            
            calculated_leftSpeed = sum(left_speeds) / list_len #len(left_speeds)
            calculated_rightSpeed = sum(right_speeds) / list_len #len(right_speeds)

            Motors.set_motor_speed(calculated_leftSpeed, calculated_rightSpeed)

            Draw.draw_speedometer(frame, 55, frame_height - 100, calculated_leftSpeed, label="Left")
            Draw.draw_speedometer(frame, frame_width-55, frame_height - 100, calculated_rightSpeed, label="Right")    

            # Convert speed lists to string format
            left_speeds_text = "Left: " + ", ".join(map(str, left_speeds))
            right_speeds_text = "Right: " + ", ".join(map(str, right_speeds))
            # Display the lists on the frame
            cv2.putText(frame, left_speeds_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, right_speeds_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Store the processed frame in a thread-safe way
            with frame_lock:
                processed_frame = frame.copy()
            
            # Display the frame with bounding boxes and labels
            #cv2.imshow("Detection", frame)

            # Exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                Motors.clean()
                break
        else:
            # Prevent 'Target is Lost' message when Pursuit is re-enabled
            last_seen_time = time.time()
            if not manual_mode:
                Motors.clean()
            # if manual_mode:
            #     Motors.set_motor_speed(40, 40)
                       
except KeyboardInterrupt:
    print("Program terminated by user.")
    Motors.clean()

except Exception as e:
    print(f"Error: {e}")
    Motors.clean()


finally:
    print("Cleaning up GPIO...")
    GPIO.cleanup()
    #cap.release()
    cv2.destroyAllWindows()

