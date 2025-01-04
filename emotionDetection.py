
# Emotion detection using DEEPFACE.

from facenet_pytorch import MTCNN
import cv2
import torch
import time
import os
from deepface import DeepFace  # Import DeepFace for emotion recognition

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize MTCNN model for face detection
mtcnn = MTCNN(keep_all=True, device=device)

# Start webcam
cap = cv2.VideoCapture(0)

# Create folder to save images if it doesn't exist
save_folder = "saved_faces"
os.makedirs(save_folder, exist_ok=True)

face_id = 0  # Variable to save images with unique filenames
max_faces_to_save = 50  # Maximum number of faces to save
saved_faces_count = 0  # Counter to track how many faces have been saved
last_saved_time = time.time()  # Track the last time a face was saved
time_interval = 2  # Time interval (in seconds) to save a new face

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break
    
    # Detect faces using MTCNN
    boxes, probs = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            # Extract the face region
            face = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            
            # Use DeepFace to analyze emotions
            result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']
            
            # Annotate the frame with the detected emotion
            cv2.putText(frame, f"{dominant_emotion}", 
                        (int(box[0]), int(box[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save the face if conditions are met
            current_time = time.time()
            if current_time - last_saved_time > time_interval and saved_faces_count < max_faces_to_save:
                face_filename = os.path.join(save_folder, f"face_{face_id}.jpg")
                cv2.imwrite(face_filename, face)
                saved_faces_count += 1
                face_id += 1
                last_saved_time = current_time

            # Draw bounding boxes around faces
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3)
    
    # Display the frame with face detection and emotion
    cv2.imshow('MTCNN Face Detection with Emotion', frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
