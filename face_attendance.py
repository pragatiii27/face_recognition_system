import face_recognition
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime

# Path to folder with known faces (add images named as person's name)
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

def main():
    print("[INFO] Loading known faces...")
    known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR)
    print(f"[INFO] {len(known_names)} faces loaded: {known_names}")

    video_capture = cv2.VideoCapture(0)
    process_every_n_frame = 2
    frame_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Only process every Nth frame for performance
        if frame_count % process_every_n_frame == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    best_match_index = np.argmin(face_distances)
                    name = known_names[best_match_index]
                    mark_attendance(name)

                # Draw rectangle and label
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom-20), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (left+6, bottom-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        frame_count += 1
        cv2.imshow('Face Recognition Attendance', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()