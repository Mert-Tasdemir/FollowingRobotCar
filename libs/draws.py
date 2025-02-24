import cv2
import numpy as np
import math

MAX_SPEED = 100.0

def draw_target(frame, target_size, target_color):
    x1, y1, x2, y2 = target_size
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
    """
    alpha = 0.2  # Transparency factor for darkening
    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    blurred_overlay = cv2.GaussianBlur(blended, (7, 7), 0)
    img[mask[:, :, 0] == 255] = blurred_overlay[mask[:, :, 0] == 255]
    """
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
