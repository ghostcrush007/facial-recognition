import cv2
import face_recognition
import os
import pandas as pd
from tkinter import Tk, Button, messagebox
import threading
import atexit

# Load user details from CSV
user_data_file = " the folder path where you want to store the user data ex. user_data.csv "
user_data_df = pd.read_csv(user_data_file)

# Path to known faces
known_faces_dir = "saved_faces/"  # folder name where the uploaded images are stored

known_face_encodings = []
known_face_names = []
known_face_ticket_types = {}

# Load and encode known faces
print("[INFO] Loading known faces...")
for file_name in os.listdir(known_faces_dir):
    file_path = os.path.join(known_faces_dir, file_name)
    
    if file_name.endswith((".jpg", ".jpeg", ".png")):
        try:
            image = face_recognition.load_image_file(file_path)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                known_face_encodings.append(encodings[0])
                name = os.path.splitext(file_name)[0]
                known_face_names.append(name)
                
                # Fetch ticket type from CSV
                user_row = user_data_df[user_data_df['Name'] == name]
                ticket_type = user_row['Ticket_Type'].values[0] if not user_row.empty else "Unknown"
                
                known_face_ticket_types[name] = ticket_type
                print(f"[INFO] Encoding stored for: {name} (Ticket: {ticket_type})")
            else:
                print(f"[WARNING] No face found in: {file_name}")
        except Exception as e:
            print(f"[ERROR] Failed to process {file_name}: {e}")

print(f"[INFO] Faces loaded: {len(known_face_encodings)}")

# Initialize webcam
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not video_capture.isOpened():
    print("[ERROR] Cannot access webcam.")
    exit()

# Ensure release on exit
atexit.register(lambda: video_capture.release())

open_windows = {}

# Show user info pop-up
def show_user_info(user_name):
    user_row = user_data_df[user_data_df['Name'] == user_name]
    
    if not user_row.empty:
        email = user_row['Email'].values[0]
        address = user_row['Address'].values[0]
        ticket_type = user_row['Ticket_Type'].values[0]
        
        info = (
            f"Name: {user_name}\n"
            f"Email: {email}\n"
            f"Address: {address}\n"
            f"Ticket Type: {ticket_type}"
        )
        messagebox.showinfo(f"{user_name} - User Details", info)
    else:
        messagebox.showwarning("User Info", f"No information available for {user_name}.")
    
    open_windows.pop(user_name, None)

# Create info button
def create_info_button(user_name, x, y):
    if user_name not in open_windows:
        open_windows[user_name] = True
        window = Tk()
        window.title(f"{user_name} - User Details")
        
        button = Button(window, text=f"View {user_name} Info", command=lambda: show_user_info(user_name))
        button.pack(pady=20)
        
        window.geometry(f"200x100+{x}+{y}")
        window.mainloop()

# Main loop
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("[ERROR] Failed to capture frame.")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    recognized_faces = []
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"
        ticket_type = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]
            ticket_type = known_face_ticket_types.get(name, "Unknown")

        top, right, bottom, left = [v * 2 for v in face_location]
        
        # Draw rectangle and display name + ticket type
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} - {ticket_type}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        recognized_faces.append((name, left + 20, top - 40))

    if recognized_faces:
        cv2.putText(frame, "Press 'Space' for info", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    if cv2.waitKey(1) & 0xFF == ord(' '):
        for user_name, x, y in recognized_faces:
            threading.Thread(target=create_info_button, args=(user_name, x, y), daemon=True).start()

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
