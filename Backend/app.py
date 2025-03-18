from flask import Flask, Response, jsonify
from flask_cors import CORS, cross_origin
import cv2
import face_recognition
import numpy as np
import sys
import os

app = Flask(__name__)
CORS(app)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Database.db_connection import connect_db  

# Load student face encodings
def load_students_from_db():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT student_id, name, date_of_birth, major, face_embedding FROM public.students;")
    students = cursor.fetchall()
    cursor.close()
    conn.close()

    known_encodings = []
    student_data = {}

    for student_id, name, date_of_birth, major, stored_embedding_str in students:
        encoding = np.array(list(map(float, stored_embedding_str.split(","))))  
        known_encodings.append(encoding)
        student_data[student_id] = {"name": name, "dob": date_of_birth, "major": major}

    return known_encodings, student_data

known_encodings, student_data = load_students_from_db()
camera = cv2.VideoCapture(0)  # Use default webcam

last_recognized_student = None

def generate_frames():
    global last_recognized_student

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            recognized_student = None  # Track recognized student

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)

                if True in matches:
                    matched_idx = matches.index(True)
                    student_id = list(student_data.keys())[matched_idx]
                    student_info = student_data[student_id]
                    recognized_student = student_id  # Store recognized student

                    # ✅ Draw Green Rectangle for Recognized Face
                    top, right, bottom, left = face_location
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, student_info["name"], (left, top - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    # ❌ Draw Red Rectangle for Unknown Face
                    top, right, bottom, left = face_location
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (left, top - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if recognized_student:
                last_recognized_student = recognized_student
            elif face_encodings:  
                last_recognized_student = "unknown"  # Store "unknown" if face is detected but not recognized
            else:
                last_recognized_student = None  # No face detected

            # Encode frame for streaming
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_student_info')
@cross_origin()
def get_student_info():
    global last_recognized_student

    if last_recognized_student in student_data:
        return jsonify({last_recognized_student: student_data[last_recognized_student]})  
    elif last_recognized_student == "unknown":
        return jsonify({"status": "unknown"})  # Send "unknown" if face detected but not recognized
    else:
        return jsonify({"status": "waiting"})  # Send "waiting" if no face is detected

if __name__ == "__main__":
    app.run(debug=True)
