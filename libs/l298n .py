import RPi.GPIO as GPIO

# Define motor control pins (Raspberry Pi GPIO pins)
LEFT_IN1    = 23  # Left motor direction pin 1 (connect to IN1 on L298N)
LEFT_IN2    = 24  # Left motor direction pin 2 (connect to IN2 on L298N)
EN_A        = 25  # Left motor speed control (PWM) (connect to ENA on L298N)

RIGHT_IN3   = 22  # Right motor direction pin 1 (connect to IN3 on L298N)
RIGHT_IN4   = 27  # Right motor direction pin 2 (connect to IN4 on L298N)
EN_B        = 17  # Right motor speed control (PWM) (connect to ENB on L298N)

FREQUENCY   = 100 # 100 Hz frequency


def init_rower():
    # Initialize GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    # Set up motor pins as outputs
    GPIO.setup([LEFT_IN1, LEFT_IN2, RIGHT_IN3, RIGHT_IN4, EN_A, EN_B], GPIO.OUT)

    # Set up PWM for motor speed control
    left_pwm = GPIO.PWM(EN_A, FREQUENCY)
    right_pwm = GPIO.PWM(EN_B, FREQUENCY)
