import face_recognition
import numpy as np
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Database.db_connection import connect_db   # Import connection function

def encode_face(image_path, student_id, name, DOB, major):
    # Load image
    image = face_recognition.load_image_file(image_path)

    # Get face encoding
    encodings = face_recognition.face_encodings(image)
    if len(encodings) == 0:
        print("❌ No face detected! Provide a clear image.")
        return

    encoding_str = ",".join(map(str, encodings[0]))  # Convert NumPy array to string

    # Connect to database
    conn = connect_db()
    if conn is None:
        print("❌ Could not connect to database.")
        return
    cursor = conn.cursor()

    try:
        # Check if student ID already exists
        cursor.execute("SELECT student_id FROM public.students WHERE student_id = %s", (student_id,))
        
        if cursor.fetchone():
            print(f"⚠️ Student ID {student_id} already exists.")
            return

        # Insert student into database
        cursor.execute("""
            INSERT INTO students (name, student_id, date_of_birth, major, face_embedding) 
            VALUES (%s, %s, %s, %s, %s)
        """, (name, student_id, DOB, major, encoding_str))

        conn.commit()
        print(f"✅ Student {name} added successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        conn.rollback()

    finally:
        cursor.close()
        conn.close()

# Add a student
encode_face("images/S12340.jpg", "S12340", "Abraham Moncherry", "1999-01-01", "Computer Science")
encode_face("images/S12341.jpg", "S12341", "Mohamad Yousouf", "1990-03-2", "Chemical Engineering")
encode_face("images/S12342.jpg", "S12342", "Shahmeen Ganesh", "2000-12-15", "Civil Engineering")
encode_face("images/S12343.jpg", "S12343", "Puhui Lam", "2001-01-5", "Computer Science")
encode_face("images/S12344.jpg", "S12344", "Elsie Ok", "1999-09-1", "Computer Science")
encode_face("images/S12345.jpg", "S12345", "Jake Gyllenhaal", "1980-12-19", "Pyschology")
encode_face("images/S12346.jpg", "S12346", "Daniel Day-Lewis", "1957-04-29", "Drama")
encode_face("images/S12347.jpg", "S12347", "Aubrey Drake Graham", "1986-10-24", "Music Production")
encode_face("images/S12348.jpg", "S12348", "Beyoncé", "1981-09-4", "Economics")
encode_face("images/S12349.jpg", "S12349", "Amy Adams", "1974-08-20", "Polical Science")

