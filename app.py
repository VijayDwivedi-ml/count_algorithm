import cv2
import streamlit as st
import numpy as np

cap = cv2.VideoCapture('part_000.mp4')
count = 0
pallet_count = 0  # separate counter for pallets

params = cv2.SimpleBlobDetector_Params()

params.minThreshold = 10
params.maxThreshold = 200
params.filterByArea = True
params.minArea = 500
params.maxArea = 2000
params.filterByCircularity = False
params.filterByConvexity = True
params.minConvexity = 0.5
params.filterByInertia = True
params.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector_create(params)

# Pallet-specific blob detector (keep conveyor detector unchanged)
params_pallet = cv2.SimpleBlobDetector_Params()
params_pallet.minThreshold = 10
params_pallet.maxThreshold = 200
params_pallet.filterByArea = True
params_pallet.minArea = 5000
params_pallet.maxArea = 50000
params_pallet.filterByCircularity = False
params_pallet.filterByConvexity = True
params_pallet.minConvexity = 0.5
params_pallet.filterByInertia = True
params_pallet.minInertiaRatio = 0.01
pallet_detector = cv2.SimpleBlobDetector_create(params_pallet)

# Conveyor ROI (rotated)
roi_x, roi_y, roi_w, roi_h = 350, 125, 200, 200

# Pallet ROI (fixed)
pallet_x, pallet_y, pallet_w, pallet_h = 500, 475, 250, 200

rotation_angle = st.sidebar.slider("Rotation Angle", -180, 180, -30, 1)

counting_line_y = roi_h // 2

tracked_objects = []
pallet_tracked_objects = []  # separate tracking for pallets

max_track_frames = 5

st.title("Box & Pallet Counter")
frame_window = st.empty()
counter_window = st.empty()

def rotate_point(point, center, angle_degrees):
    angle_rad = np.radians(angle_degrees)
    x, y = point
    cx, cy = center
    x_translated = x - cx
    y_translated = y - cy
    x_rotated = x_translated * np.cos(angle_rad) - y_translated * np.sin(angle_rad)
    y_rotated = x_translated * np.sin(angle_rad) + y_translated * np.cos(angle_rad)
    x_final = x_rotated + cx
    y_final = y_rotated + cy
    return (int(x_final), int(y_final))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === Conveyor logic (unchanged) ===
    roi_center_x = roi_x + roi_w // 2
    roi_center_y = roi_y + roi_h // 2
    corners = [
        (roi_x, roi_y),
        (roi_x + roi_w, roi_y),
        (roi_x + roi_w, roi_y + roi_h),
        (roi_x, roi_y + roi_h)
    ]
    rotated_corners = [rotate_point(corner, (roi_center_x, roi_center_y), rotation_angle) for corner in corners]
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [np.array(rotated_corners)], (255, 255, 255))
    rotated_roi = cv2.bitwise_and(frame, mask)
    rotated_roi = rotated_roi[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    gray = cv2.cvtColor(rotated_roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    keypoints = detector.detect(thresh)

    current_keypoints = []
    for keypoint in keypoints:
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        orig_x = roi_x + x
        orig_y = roi_y + y
        current_keypoints.append((orig_x, orig_y))

    for obj in tracked_objects:
        obj['found'] = False
        obj['age'] += 1

    for x, y in current_keypoints:
        matched = False
        for obj in tracked_objects:
            if not obj['found']:
                last_x, last_y = obj['positions'][-1]
                distance = np.sqrt((x - last_x)**2 + (y - last_y)**2)
                if distance < 30:
                    obj['positions'].append((x, y))
                    obj['found'] = True
                    obj['age'] = 0
                    rotated_point = rotate_point((x, y), (roi_center_x, roi_center_y), -rotation_angle)
                    current_state = 'below' if rotated_point[1] > roi_center_y else 'above'
                    if (obj['state'] == 'above' and current_state == 'below' and not obj['counted']):
                        count += 1
                        obj['counted'] = True
                    obj['state'] = current_state
                    matched = True
                    break
        if not matched:
            rotated_point = rotate_point((x, y), (roi_center_x, roi_center_y), -rotation_angle)
            new_state = 'below' if rotated_point[1] > roi_center_y else 'above'
            tracked_objects.append({
                'positions': [(x, y)],
                'state': new_state,
                'counted': new_state == 'below',
                'found': True,
                'age': 0
            })
            if new_state == 'below':
                count += 1

    tracked_objects = [obj for obj in tracked_objects if obj['age'] < max_track_frames]

    # Draw conveyor ROI and line
    for i in range(4):
        pt1 = rotated_corners[i]
        pt2 = rotated_corners[(i+1) % 4]
        cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
    line_start = rotate_point((roi_x, roi_center_y), (roi_center_x, roi_center_y), rotation_angle)
    line_end = rotate_point((roi_x + roi_w, roi_center_y), (roi_center_x, roi_center_y), rotation_angle)
    cv2.line(frame, line_start, line_end, (0, 255, 0), 2)

    # === Pallet logic with crossing line ===
    pallet_roi = frame[pallet_y:pallet_y+pallet_h, pallet_x:pallet_x+pallet_w]
    gray_pallet = cv2.cvtColor(pallet_roi, cv2.COLOR_BGR2GRAY)
    _, thresh_pallet = cv2.threshold(gray_pallet, 127, 255, cv2.THRESH_BINARY)
    pallet_keypoints = pallet_detector.detect(thresh_pallet)

    current_pallet_keypoints = []
    for keypoint in pallet_keypoints:
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        orig_x = pallet_x + x
        orig_y = pallet_y + y
        current_pallet_keypoints.append((orig_x, orig_y))

    pallet_center_y = pallet_y + pallet_h // 2

    for obj in pallet_tracked_objects:
        obj['found'] = False
        obj['age'] += 1

    for x, y in current_pallet_keypoints:
        matched = False
        for obj in pallet_tracked_objects:
            if not obj['found']:
                last_x, last_y = obj['positions'][-1]
                distance = np.sqrt((x - last_x)**2 + (y - last_y)**2)
                if distance < 50:
                    obj['positions'].append((x, y))
                    obj['found'] = True
                    obj['age'] = 0
                    current_state = 'below' if y > pallet_center_y else 'above'
                    if (obj['state'] == 'above' and current_state == 'below' and not obj['counted']):
                        pallet_count += 1
                        obj['counted'] = True
                    obj['state'] = current_state
                    matched = True
                    break
        if not matched:
            new_state = 'below' if y > pallet_center_y else 'above'
            pallet_tracked_objects.append({
                'positions': [(x, y)],
                'state': new_state,
                'counted': new_state == 'below',
                'found': True,
                'age': 0
            })
            if new_state == 'below':
                pallet_count += 1

    pallet_tracked_objects = [obj for obj in pallet_tracked_objects if obj['age'] < max_track_frames]

    # Draw pallet ROI and center line
    cv2.rectangle(frame, (pallet_x, pallet_y), (pallet_x+pallet_w, pallet_y+pallet_h), (255, 0, 0), 2)
    cv2.line(frame, (pallet_x, pallet_center_y), (pallet_x+pallet_w, pallet_center_y), (0, 255, 255), 2)

    # Display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame)
    counter_window.write(f"Boxes Count: {count} | Pallets Count: {pallet_count}")

cap.release()
