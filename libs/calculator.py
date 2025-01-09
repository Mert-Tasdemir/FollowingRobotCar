STACK_LEN = 10

MAX_TARGET = 0.60
MIN_TARGET = 0.08
SCOPE_TARGET = 1.0 / (MAX_TARGET - MIN_TARGET)

DSLIP_X = 0.90
SCOPE_SLIP_X = 1.0 / DSLIP_X

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
