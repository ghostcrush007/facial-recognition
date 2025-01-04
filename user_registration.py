import streamlit as st
import cv2
import face_recognition
import numpy as np
from PIL import Image
import os
import pandas as pd

# Function to register user and save their face encoding
def register_user(name, email, address, ticket_type, img_bgr):
    # Save the user image and details to a CSV file or database
    user_image_path = os.path.join("saved_faces", f"{name}.jpg")
    
    # Save user details with image path
    user_data = {'Name': [name], 'Email': [email], 'Address': [address], 'Ticket_Type': [ticket_type], 'Image_Path': [user_image_path]}
    df = pd.DataFrame(user_data)
    
    # Save user details to CSV
    user_data_file = "user_data.csv"  # You can change the file name/path as required
    if os.path.exists(user_data_file):
        df.to_csv(user_data_file, mode='a', header=False, index=False)
    else:
        df.to_csv(user_data_file, mode='w', header=True, index=False)

    # Generate the face encoding
    face_encoding = face_recognition.face_encodings(img_bgr)[0]

    # Save the image in the known faces folder
    known_faces_dir = "saved_faces/"
    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)

    # Save the image with the name as the user's name
    cv2.imwrite(user_image_path, img_bgr)
    
    # Save the encoding to a file or database
    encoding_file = "known_face_encodings.npy"
    
    if os.path.exists(encoding_file):
        # Load the existing encodings and append the new one
        known_face_encodings = np.load(encoding_file, allow_pickle=True)
        # Append the new encoding (convert it into an array if it is a list)
        known_face_encodings = np.append(known_face_encodings, [face_encoding], axis=0)
    else:
        # If no encodings exist, create a new array
        known_face_encodings = np.array([face_encoding])

    # Save the updated encodings back to the file
    np.save(encoding_file, known_face_encodings)
    
    st.success(f"User {name} registered successfully!")

# User details input form
st.subheader("User Registration")

# User input fields
name = st.text_input("Enter your name")
email = st.text_input("Enter your email")
address = st.text_input("Enter your address")
ticket_type = st.selectbox("Select your ticket type", ["Normal", "High", "VIP"])

# Option to upload image
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if name and email and address and ticket_type:
    if uploaded_image is not None:
        # Convert the uploaded image to a format that can be processed
        pil_image = Image.open(uploaded_image)
        img = np.array(pil_image)

        # Convert PIL image to BGR (OpenCV format)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        face_locations = face_recognition.face_locations(img_bgr)

        if len(face_locations) > 0:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
            st.write(f"Found {len(face_locations)} face(s) in the image.")

            # Register button
            if st.button("Register User"):
                register_user(name, email, address, ticket_type, img_bgr)

                # Reset fields after registration
                name = ""
                email = ""
                address = ""
                ticket_type = "Normal"
                uploaded_image = None
        else:
            st.warning("No faces detected in the uploaded image. Please try again with a clear face image.")
    else:
        st.warning("Please upload an image to complete registration.")
else:
    st.warning("Please enter all the required details (name, email, address, and ticket type).")
