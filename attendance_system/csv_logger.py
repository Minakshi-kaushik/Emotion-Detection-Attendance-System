import os
import pandas as pd
from datetime import datetime

CSV_FILE = os.path.join(os.path.dirname(__file__), "attendance.csv")


def create_attendance_file():
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame(columns=["Name", "Date", "Time", "Status"])
        df.to_csv(CSV_FILE, index=False)
        print("Attendance file created.")


def mark_attendance(name):
    create_attendance_file()

    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")

    current_time = now.strftime("%H:%M:%S")

    df = pd.read_csv(CSV_FILE)

    already_marked = ((df["Name"] == name) & (df["Date"] == current_date)).any()

    if already_marked:
        print(f"{name} already marked today.")
        return

    new_record = pd.DataFrame(
        {
            "Name": [name],
            "Date": [current_date],
            "Time": [current_time],
            "Status": ["Present"],
        }
    )

    df = pd.concat([df, new_record], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

    print(f"Attendance marked for {name}.")


if __name__ == "__main__":
    mark_attendance("Minakshi")
    mark_attendance("Student1")
    mark_attendance("Student2")
    mark_attendance("Student3")
