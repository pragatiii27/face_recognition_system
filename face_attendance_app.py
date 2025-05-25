import streamlit as st
import face_recognition
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime

KNOWN_FACES_DIR = 'known_faces'
ATTENDANCE_CSV = 'attendance.csv'

def load_known_faces(known_faces_dir):
    known_encodings = []
    known_names = []
    for filename in os.listdir(known_faces_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = face_recognition.load_image_file(f"{known_faces_dir}/{filename}")
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_encodings.append(encoding[0])
                known_names.append(os.path.splitext(filename)[0])
    return known_encodings, known_names

def mark_attendance(name):
    now = datetime.now()
    dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
    if not os.path.exists(ATTENDANCE_CSV):
        df = pd.DataFrame(columns=['Name', 'Time'])
        df.to_csv(ATTENDANCE_CSV, index=False)
    df = pd.read_csv(ATTENDANCE_CSV)
    if ((df['Name'] == name) & (df['Time'].str[:10] == dt_string[:10])).any():
        return  # Already marked for today
    new_entry = pd.DataFrame({'Name': [name], 'Time': [dt_string]})
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(ATTENDANCE_CSV, index=False)

st.title("Face Recognition Attendance System")

st.write("Upload a photo or take a snapshot to mark your attendance.")

known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR)

uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "png"])
if uploaded_file is not None:
    image = face_recognition.load_image_file(uploaded_file)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    names = []
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            best_match_index = np.argmin(face_distances)
            name = known_names[best_match_index]
            mark_attendance(name)
        names.append(name)
        # Draw rectangle and label
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left+6, bottom-6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    st.image(frame, channels="BGR", caption="Processed Image")
    if names:
        st.success(f"Recognized: {', '.join(names)}")
    else:
        st.warning("No known faces recognized.")

if st.button("Show Attendance Log"):
    if os.path.exists(ATTENDANCE_CSV):
        df = pd.read_csv(ATTENDANCE_CSV)
        st.dataframe(df)
    else:
        st.info("No attendance has been marked yet.")