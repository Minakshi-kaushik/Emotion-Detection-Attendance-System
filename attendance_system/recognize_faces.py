import cv2
import os
from csv_logger import mark_attendance

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths
TRAINER_PATH = os.path.join(BASE_DIR, "trainer.yml")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

# Load trained recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(TRAINER_PATH)

# Load labels
labels = {}

with open(LABELS_PATH, "r") as f:
    for line in f:
        label_id, name = line.strip().split(",")
        labels[int(label_id)] = name

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Start webcam
cap = cv2.VideoCapture(0)

# Prevent repeated attendance marking
recognized_today = set()

print("Face Recognition System Started")
print("Press 'Q' to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to access webcam")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50)
    )

    for x, y, w, h in faces:
        face_roi = gray[y : y + h, x : x + w]

        try:
            label, confidence = recognizer.predict(face_roi)

            # Lower confidence = better match
            if confidence < 80:
                name = labels.get(label, "Unknown")

                if name not in recognized_today:
                    mark_attendance(name)
                    recognized_today.add(name)

            else:
                name = "Unknown"

        except:
            name = "Unknown"
            confidence = 0

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display name and confidence
        cv2.putText(
            frame,
            f"{name} ({confidence:.0f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Attendance Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
