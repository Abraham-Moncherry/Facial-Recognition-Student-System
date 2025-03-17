from flask import Flask, Response, jsonify
import cv2
import face_recognition
import numpy as np
import sys
import os

# Add project root to sys.path to access Database folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import connect_db function from db_connection.py
from Database.db_connection import connect_db  

app = Flask(__name__)

# Load stored student face encodings
def get_students_from_db():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT student_id, name, major, face_embedding FROM public.students;")
    students = cursor.fetchall()
    cursor.close()
    conn.close()

    known_encodings = []
    student_data = {}

    for student_id, name, major, stored_embedding_str in students:
        encoding = np.array(list(map(float, stored_embedding_str.split(","))))  
        known_encodings.append(encoding)
        student_data[student_id] = {"name": name, "major": major}

    return known_encodings, student_data

# Start webcam
camera = cv2.VideoCapture(0)  # Use 0 for default camera

def generate_frames():
    known_encodings, student_data = get_students_from_db()

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            student_info = None
            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                if True in matches:
                    matched_idx = matches.index(True)
                    student_id = list(student_data.keys())[matched_idx]
                    student_info = student_data[student_id]

                    # Draw rectangle around recognized face
                    top, right, bottom, left = face_location
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, student_info["name"], (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Encode frame for streaming
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# API to stream video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# API to get recognized student info
@app.route('/get_student_info')
def get_student_info():
    known_encodings, student_data = get_students_from_db()
    return jsonify(student_data)

if __name__ == "__main__":
    app.run(debug=True)
