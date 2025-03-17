import cv2
import face_recognition
import numpy as np
import psycopg2
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Database.db_connection import connect_db  # Import DB connection

# Load environment variables
load_dotenv()

def recognize_student():
    """Uses webcam to recognize students based on stored face embeddings."""

    # Connect to database
    conn = connect_db()
    if conn is None:
        print("❌ Could not connect to database.")
        return

    cursor = conn.cursor()

    try:
        # ✅ Fetch all students' embeddings from the database
        cursor.execute("SELECT student_id, name, major, face_embedding FROM public.students;")
        students = cursor.fetchall()

        if not students:
            print("❌ No students found in the database!")
            return

        # Convert stored embeddings to NumPy format
        known_student_ids = []
        known_names = []
        known_majors = []
        known_encodings = []

        for student in students:
            student_id, name, major, stored_embedding_str = student
            stored_embedding = np.array([float(x) for x in stored_embedding_str.split(",")])

            known_student_ids.append(student_id)
            known_names.append(name)
            known_majors.append(major)
            known_encodings.append(stored_embedding)

        # ✅ Open Webcam
        video_capture = cv2.VideoCapture(0)

        print("🎥 Webcam started. Press 'q' to quit.")

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("❌ Failed to capture image from webcam.")
                break

            # Convert frame from BGR (OpenCV format) to RGB (face_recognition format)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect face locations & encodings
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                name = "Unknown"
                student_id = None
                major = None

                if True in matches:
                    match_index = matches.index(True)
                    student_id = known_student_ids[match_index]
                    name = known_names[match_index]
                    major = known_majors[match_index]

                # Draw rectangle around face & display student info
                top, right, bottom, left = face_location
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green for recognized, Red for unknown
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, f"{name} ({student_id}) - {major}", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Show the video frame
            cv2.imshow("Student Recognition System", frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        video_capture.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        cursor.close()
        conn.close()


# Run the webcam-based recognition
recognize_student()
