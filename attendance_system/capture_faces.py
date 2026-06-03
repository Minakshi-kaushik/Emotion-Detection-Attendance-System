import cv2
import os
import time

face_cascade = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml")

save_path = "students/student3"

if not os.path.exists(save_path):
    os.makedirs(save_path)

cap = cv2.VideoCapture(0)

count = 0
max_images = 50

last_capture = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        current_time = time.time()

        if current_time - last_capture > 2:
            face = gray[y : y + h, x : x + w]

            count += 1

            cv2.imwrite(f"{save_path}/{count}.jpg", face)

            last_capture = current_time

    cv2.putText(
        frame,
        f"Images: {count}/{max_images}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Capture Faces", frame)

    if count >= max_images:
        print("Dataset collection completed!")
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
