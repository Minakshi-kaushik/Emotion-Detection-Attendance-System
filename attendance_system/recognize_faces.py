import cv2
import os
import numpy as np

from csv_logger import mark_attendance

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense,
)

# ==========================
# BASE DIRECTORY
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================
# FILE PATHS
# ==========================
TRAINER_PATH = os.path.join(BASE_DIR, "trainer.yml")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, "..", "model.weights.h5")

# ==========================
# LOAD FACE RECOGNIZER
# ==========================
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(TRAINER_PATH)

# ==========================
# LOAD LABELS
# ==========================
labels = {}

with open(LABELS_PATH, "r") as f:
    for line in f:
        label_id, name = line.strip().split(",")
        labels[int(label_id)] = name

# ==========================
# LOAD HAAR CASCADE
# ==========================
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


# ==========================
# CNN BLOCKS
# ==========================
def Convolution(input_tensor, filters, kernel_size):

    x = Conv2D(filters=filters, kernel_size=kernel_size, padding="same")(input_tensor)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    return x


def Dense_f(input_tensor, nodes):

    x = Dense(nodes)(input_tensor)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.25)(x)

    return x


# ==========================
# BUILD EMOTION MODEL
# ==========================
def build_emotion_model():

    inputs = Input(shape=(48, 48, 1))

    conv_1 = Convolution(inputs, 32, (3, 3))
    conv_2 = Convolution(conv_1, 64, (5, 5))
    conv_3 = Convolution(conv_2, 128, (3, 3))

    flatten = Flatten()(conv_3)

    dense_1 = Dense_f(flatten, 256)

    output = Dense(7, activation="softmax")(dense_1)

    model = Model(inputs=inputs, outputs=output)

    return model


# ==========================
# LOAD EMOTION MODEL
# ==========================
emotion_model = build_emotion_model()
emotion_model.load_weights(MODEL_WEIGHTS_PATH)

print("Emotion model loaded successfully.")

EMOTIONS_LIST = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]

# ==========================
# START CAMERA
# ==========================
cap = cv2.VideoCapture(0)

recognized_today = set()

print("Emotion-Aware Attendance System Started")
print("Press 'Q' to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to access webcam")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(50, 50),
    )

    for x, y, w, h in faces:
        face_roi = gray[y : y + h, x : x + w]

        try:
            # ==========================
            # FACE RECOGNITION
            # ==========================
            label, confidence = recognizer.predict(face_roi)

            if confidence < 80:
                name = labels.get(label, "Unknown")
            else:
                name = "Unknown"

            # ==========================
            # EMOTION DETECTION
            # ==========================
            emotion_face = cv2.resize(face_roi, (48, 48))

            emotion_face = emotion_face.reshape(1, 48, 48, 1)

            prediction = emotion_model.predict(emotion_face, verbose=0)

            print(
                {
                    emotion_name: round(float(score), 3)
                    for emotion_name, score in zip(EMOTIONS_LIST, prediction[0])
                }
            )

            emotion = EMOTIONS_LIST[np.argmax(prediction)]

            # ==========================
            # MARK ATTENDANCE
            # ==========================
            if name != "Unknown" and name not in recognized_today:
                mark_attendance(name, emotion)

                recognized_today.add(name)

        except Exception as e:
            print("Error:", e)

            name = "Unknown"
            emotion = "Unknown"
            confidence = 0

        # ==========================
        # DRAW FACE BOX
        # ==========================
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ==========================
        # DISPLAY NAME + EMOTION
        # ==========================
        cv2.putText(
            frame,
            f"{name} | {emotion}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Emotion-Aware Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
