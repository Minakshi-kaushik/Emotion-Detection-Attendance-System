import os

base_path = "students"

total_images = 0

for person in os.listdir(base_path):
    person_path = os.path.join(base_path, person)

    if os.path.isdir(person_path):
        count = len([f for f in os.listdir(person_path) if f.endswith(".jpg")])

        total_images += count
        print(f"{person}: {count} images")

print("\nTotal Images:", total_images)
