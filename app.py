import cv2
import streamlit as st
import numpy as np
import tempfile
import os
from PIL import Image
import io

# File uploader for video
uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # Save uploaded file to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_path = tmp_file.name
    
    cap = cv2.VideoCapture(video_path)
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
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temporary video file for output
    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    progress_bar = st.progress(0)
    counter_display = st.empty()
    video_placeholder = st.empty()
    results_display = st.empty()
    
    current_frame = 0
    processed_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_frame += 1
        progress_bar.progress(current_frame / total_frames)

        # Store original frame for drawing
        display_frame = frame.copy()
        
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

        # Draw bounding boxes and annotations on the display frame
        cv2.rectangle(display_frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0, 0, 255), 2)
        cv2.line(display_frame, (roi_x, roi_y + counting_line_y), (roi_x + roi_w, roi_y + counting_line_y), (0, 255, 0), 2)
        
        # Draw keypoints on ROI
        for kp in keypoints:
            x, y = int(kp.pt[0] + roi_x), int(kp.pt[1] + roi_y)
            cv2.circle(display_frame, (x, y), 10, (0, 0, 255), -1)
        
        # Add count text
        cv2.putText(display_frame, f"Count: {count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write frame to output video
        out.write(display_frame)
        
        # Convert to RGB for Streamlit display (optional preview)
        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        processed_frames.append(display_frame_rgb)
        
        # Update counter display
        counter_display.write(f"**Current Count: {count}**")
        
        # Show preview of current frame (optional, can be heavy)
        if current_frame % 30 == 0:  # Show every 30th frame to reduce load
            video_placeholder.image(display_frame_rgb, caption=f"Frame {current_frame}")

    cap.release()
    out.release()
    
    # Display the processed video with bounding boxes
    st.success("✅ Processing complete!")
    st.write(f"**Final Count: {count} boxes detected**")
    
    # Read the processed video file and display it
    with open(output_video_path, 'rb') as video_file:
        video_bytes = video_file.read()
    
    st.video(video_bytes)
    
    # Clean up temporary files
    os.unlink(video_path)
    os.unlink(output_video_path)
    
else:
    st.info("Please upload a video file to begin counting")
