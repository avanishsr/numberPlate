import os
import cv2
import torch
import base64
import re
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import easyocr
from io import BytesIO
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv8 model (ensure the best.pt is in the same folder as this script)
model = YOLO("weights/best.pt")

# Create EasyOCR reader (for OCR)
reader = easyocr.Reader(['en'])

# Ensure the output folder exists
if not os.path.exists("static/output"):
    os.makedirs("static/output")


# Home Route to upload video
@app.route('/')
def home():
    return render_template('index.html')


# Route to process the video and detect the best license plate
@app.route('/detect_video', methods=['POST'])
def detect_best_plate_from_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'})

    # Save uploaded video file
    video_file = request.files['video']
    video_path = os.path.join('static/output', video_file.filename)
    video_file.save(video_path)

    # Process the video using OpenCV
    cap = cv2.VideoCapture(video_path)

    best_plate = None
    best_confidence = 0
    best_box = None
    best_frame = None

    frame_count = 0
    frame_skip = 5  # Process every 5th frame to save time
    x_confidence = 0.5  # Confidence threshold

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Only process every nth frame unless a plate with high confidence is found
        if frame_count % frame_skip == 0:
            # Run YOLOv8 inference on the current frame
            results = model(frame)

            # Check for detections
            if len(results) > 0 and results[0].boxes is not None:
                for i, box in enumerate(results[0].boxes.xyxy):
                    confidence = results[0].boxes.conf[i].item()  # Confidence score
                    x1, y1, x2, y2 = map(int, box.cpu().numpy())  # Bounding box

                    # Get the size of the bounding box (area)
                    box_area = (x2 - x1) * (y2 - y1)

                    # Select the largest plate with the highest confidence
                    if confidence > best_confidence and box_area > 1000:  # Adjust area threshold if needed
                        best_confidence = confidence
                        best_box = (x1, y1, x2, y2)
                        best_frame = frame

                    # If confidence is higher than the threshold, skip further frames for a while
                    if confidence >= x_confidence:
                        frame_count += frame_skip  # Skip the next few frames if confidence is high

        frame_count += 1

    # Release the video capture object
    cap.release()

    # Ensure we found a plate
    if best_box and best_frame is not None:
        # Crop the best detected license plate
        x1, y1, x2, y2 = best_box
        cropped_plate = best_frame[y1:y2, x1:x2]

        # Convert cropped image to base64
        cropped_image_pil = Image.fromarray(cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB))
        buffered = BytesIO()
        cropped_image_pil.save(buffered, format="JPEG")
        cropped_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Perform OCR on the cropped license plate
        ocr_result = reader.readtext(cropped_plate, detail=0)  # Extract only text

        # Convert OCR result to a single string, uppercase, no spaces, remove special characters
        if ocr_result:
            ocr_text = ''.join(ocr_result).upper()  # Join and convert to uppercase
            ocr_text_clean = re.sub(r'[^A-Z0-9]', '', ocr_text)  # Remove special characters

            # Apply additional logic:
            # 1. If the first two characters contain '0', replace with 'O'
            if len(ocr_text_clean) >= 2:
                if '0' in ocr_text_clean[:2]:
                    ocr_text_clean = ocr_text_clean[:2].replace('0', 'O') + ocr_text_clean[2:]

            # 2. If the third and fourth characters contain 'O', replace with '0'
            if len(ocr_text_clean) >= 4:
                if 'O' in ocr_text_clean[2:4]:
                    ocr_text_clean = ocr_text_clean[:2] + ocr_text_clean[2:4].replace('O', '0') + ocr_text_clean[4:]

        # Return the cleaned and adjusted OCR result and the corresponding cropped image
        return jsonify({
            'detected_text': ocr_text_clean,  # Return the modified OCR result
            'cropped_image_base64': cropped_base64
        })
    else:
        return jsonify({'error': 'No license plate detected'})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to port 5000 for local development
    app.run(host="0.0.0.0", port=port)
