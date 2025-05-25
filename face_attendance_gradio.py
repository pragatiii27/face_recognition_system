import gradio as gr
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
import os
from PIL import Image

KNOWN_FACES_DIR = 'known_faces'
ATTENDANCE_CSV = 'attendance.csv'

def load_known_faces():
    known_encodings = []
    known_names = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(('.jpg', '.png')):
            img = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, filename))
            encodings = face_recognition.face_encodings(img)
            if encodings:
                known_encodings.append(encodings[0])
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
        return False  # Already marked today
    new_entry = pd.DataFrame({'Name': [name], 'Time': [dt_string]})
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(ATTENDANCE_CSV, index=False)
    return True

def recognize_face(image):
    known_encodings, known_names = load_known_faces()
    img = np.array(image)
    rgb_img = img[:, :, ::-1]  # Convert BGR to RGB if needed

    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    recognized_names = []
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            best_match_index = np.argmin(face_distances)
            name = known_names[best_match_index]
            marked = mark_attendance(name)
            if marked:
                recognized_names.append(f"{name} (attendance marked)")
            else:
                recognized_names.append(f"{name} (already marked today)")
        else:
            recognized_names.append("Unknown")

        # Draw rectangle and label
        top, right, bottom, left = face_location
        img = cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        img = cv2.putText(img, name, (left+6, bottom-6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    if len(face_encodings) == 0:
        return img, "No face detected."
    return img, "Recognized: " + ", ".join(recognized_names)

def show_attendance():
    if os.path.exists(ATTENDANCE_CSV):
        df = pd.read_csv(ATTENDANCE_CSV)
        return df
    else:
        return pd.DataFrame(columns=["Name", "Time"])

with gr.Blocks() as demo:
    gr.Markdown("# 📸 Face Recognition Attendance System")
    gr.Markdown("Upload a photo or use your webcam. The app will recognize faces and mark attendance.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Photo or Use Webcam", sources=["upload", "webcam"], type="numpy")
            submit_btn = gr.Button("Recognize and Mark Attendance")
            output_image = gr.Image(label="Result", type="numpy")
            result_text = gr.Textbox(label="Recognition Result")
        with gr.Column():
            log_btn = gr.Button("Show Attendance Log")
            attendance_table = gr.Dataframe(label="Attendance Log", headers=["Name", "Time"], datatype=["str", "str"])

    submit_btn.click(recognize_face, inputs=image_input, outputs=[output_image, result_text])
    log_btn.click(show_attendance, outputs=attendance_table)

if __name__ == "__main__":
    demo.launch()