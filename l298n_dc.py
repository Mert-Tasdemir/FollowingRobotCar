# Python Script
# https://www.electronicshub.org/raspberry-pi-l298n-interface-tutorial-control-dc-motor-l298n-raspberry-pi/

import RPi.GPIO as GPIO          
from time import sleep

"""
in1 = 27
in2 = 22
en = 17

"""
temp1=1
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
LOW_PWM = 20
MEDIUM_PWM = 30
HIGH_PWM = 40

# Set up PWM for speed control
LEFT_PWM = GPIO.PWM(EN_A, FREQUENCY)  # 100 Hz PWM frequency
RIGHT_PWM = GPIO.PWM(EN_B, FREQUENCY)



LEFT_PWM.start(LOW_PWM)  # Start with 0% duty cycle
RIGHT_PWM.start(LOW_PWM)
"""
GPIO.setmode(GPIO.BCM)
GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(en,GPIO.OUT)
GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)
p=GPIO.PWM(en,1000)
"""


print("\n")
print("The default speed & direction of motor is LOW & Forward.....")
print("r-run s-stop f-forward b-backward l-low m-medium h-high e-exit")
print("\n")    

while(1):

    x = input("Enter something: ")
    
    if x=='r':
        print("run")
        if(temp1==1):
         GPIO.output(LEFT_IN1,GPIO.HIGH)
         GPIO.output(LEFT_IN2,GPIO.LOW)
         GPIO.output(RIGHT_IN3,GPIO.HIGH)
         GPIO.output(RIGHT_IN4,GPIO.LOW)
         print("forward")
         x='z'
        else:
         GPIO.output(LEFT_IN1,GPIO.LOW)
         GPIO.output(LEFT_IN2,GPIO.HIGH)
         GPIO.output(RIGHT_IN3,GPIO.LOW)
         GPIO.output(RIGHT_IN4,GPIO.HIGH)
         print("backward")
         x='z'


    elif x=='s':
        print("stop")
        GPIO.output(LEFT_IN1,GPIO.LOW)
        GPIO.output(LEFT_IN2,GPIO.LOW)
        GPIO.output(RIGHT_IN3,GPIO.LOW)
        GPIO.output(RIGHT_IN4,GPIO.LOW)
        x='z'

    elif x=='f':
        print("forward")
        GPIO.output(LEFT_IN1,GPIO.HIGH)
        GPIO.output(LEFT_IN2,GPIO.LOW)
        GPIO.output(RIGHT_IN3,GPIO.HIGH)
        GPIO.output(RIGHT_IN4,GPIO.LOW)
        temp1=1
        x='z'

    elif x=='b':
        print("backward")
        GPIO.output(LEFT_IN1,GPIO.LOW)
        GPIO.output(LEFT_IN2,GPIO.HIGH)
        GPIO.output(RIGHT_IN3,GPIO.LOW)
        GPIO.output(RIGHT_IN4,GPIO.HIGH)
        temp1=0
        x='z'

    elif x=='l':
        print("low")
        LEFT_PWM.start(LOW_PWM) 
        RIGHT_PWM.start(LOW_PWM)
        x='z'

    elif x=='m':
        print("medium")
        LEFT_PWM.start(MEDIUM_PWM) 
        RIGHT_PWM.start(MEDIUM_PWM)
        x='z'

    elif x=='h':
        print("high")
        LEFT_PWM.start(HIGH_PWM) 
        RIGHT_PWM.start(HIGH_PWM)
        x='z'
     
    
    elif x=='e':
        GPIO.cleanup()
        print("GPIO Clean up")
        break
    
    else:
        print("<<<  wrong data  >>>")
        print("please enter the defined data to continue.....")