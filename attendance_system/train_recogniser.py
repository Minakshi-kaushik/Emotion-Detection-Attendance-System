import cv2
import os
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "students")

faces = []
labels = []

label_dict = {}
current_label = 0
print(dataset_path)

for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_path):
        continue

    label_dict[current_label] = person_name

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        faces.append(img)
        labels.append(current_label)

    current_label += 1

labels = np.array(labels)

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(faces, labels)

recognizer.save("trainer.yml")

print("Model trained successfully!")

with open("labels.txt", "w") as f:
    for key, value in label_dict.items():
        f.write(f"{key},{value}\n")

print("\nLabel Mapping:")
for key, value in label_dict.items():
    print(f"{key} -> {value}")
