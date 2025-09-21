import cv2
import streamlit as st
import numpy as np


cap = cv2.VideoCapture('part_000.mp4')
count = 0

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

roi_x, roi_y, roi_w, roi_h = 350, 125, 200, 200

counting_line_y = roi_h // 2

tracked_objects = []  
max_track_frames = 5  


st.title("Box Counter")
frame_window = st.empty()
counter_window = st.empty()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    keypoints = detector.detect(thresh)

    current_keypoints = []
    for keypoint in keypoints:
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        current_keypoints.append((x, y))

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
                    matched = True
                    
                    current_state = 'below' if y > counting_line_y else 'above'
                    if (obj['state'] == 'above' and current_state == 'below' and not obj['counted']):
                        count += 1
                        obj['counted'] = True
                    obj['state'] = current_state
                    break
        
        if not matched:
            new_state = 'below' if y > counting_line_y else 'above'
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

    blobs = cv2.drawKeypoints(roi, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0, 0, 255), 2)
    cv2.line(frame, (roi_x, roi_y + counting_line_y), (roi_x + roi_w, roi_y + counting_line_y), (0, 255, 0), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame)
    counter_window.write(f"Count: {count}")

cap.release()