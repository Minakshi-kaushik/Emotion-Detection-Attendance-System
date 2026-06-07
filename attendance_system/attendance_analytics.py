import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_FILE = os.path.join(BASE_DIR, "attendance.csv")

REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(REPORT_DIR, exist_ok=True)

# =====================================
# LOAD DATA
# =====================================

if not os.path.exists(CSV_FILE):
    print("attendance.csv not found.")
    exit()

df = pd.read_csv(CSV_FILE)

if df.empty:
    print("No attendance records found.")
    exit()

# =====================================
# BASIC STATISTICS
# =====================================

total_records = len(df)

unique_students = df["Name"].nunique()

today = datetime.now().strftime("%Y-%m-%d")

today_records = df[df["Date"] == today]

present_today = today_records["Name"].nunique()

attendance_percentage = (
    present_today / unique_students * 100 if unique_students > 0 else 0
)

# =====================================
# PRINT SUMMARY
# =====================================

print("\n========== ATTENDANCE SUMMARY ==========")

print(f"Total Records      : {total_records}")
print(f"Registered Students: {unique_students}")
print(f"Present Today      : {present_today}")
print(f"Attendance %       : {attendance_percentage:.2f}%")

print("========================================\n")

# =====================================
# STUDENT ATTENDANCE COUNT
# =====================================

attendance_counts = df.groupby("Name").size().sort_values(ascending=False)

# =====================================
# EMOTION DISTRIBUTION
# =====================================

emotion_counts = df["Emotion"].value_counts()

# =====================================
# BAR CHART
# =====================================

plt.figure(figsize=(10, 6))

attendance_counts.plot(kind="bar")

plt.title("Student Attendance Count", fontsize=16, fontweight="bold")

plt.xlabel("Students")
plt.ylabel("Attendance Records")

plt.tight_layout()

attendance_chart = os.path.join(REPORT_DIR, "attendance_bar_chart.png")

plt.savefig(attendance_chart)

plt.close()

# =====================================
# PIE CHART
# =====================================

plt.figure(figsize=(8, 8))

emotion_counts.plot(kind="pie", autopct="%1.1f%%")

plt.title("Emotion Distribution", fontsize=16, fontweight="bold")

plt.ylabel("")

plt.tight_layout()

emotion_chart = os.path.join(REPORT_DIR, "emotion_pie_chart.png")

plt.savefig(emotion_chart)

plt.close()

# =====================================
# DAILY ATTENDANCE TREND
# =====================================

daily_attendance = df.groupby("Date")["Name"].nunique()

plt.figure(figsize=(10, 6))

daily_attendance.plot(marker="o", linewidth=2)

plt.title("Daily Attendance Trend", fontsize=16, fontweight="bold")

plt.xlabel("Date")
plt.ylabel("Students Present")

plt.grid(True)

plt.tight_layout()

trend_chart = os.path.join(REPORT_DIR, "attendance_trend.png")

plt.savefig(trend_chart)

plt.close()

# =====================================
# SAVE REPORT
# =====================================

report_file = os.path.join(REPORT_DIR, "attendance_report.txt")

with open(report_file, "w") as f:
    f.write("EMOTION-AWARE ATTENDANCE REPORT\n")
    f.write("=" * 40 + "\n\n")

    f.write(f"Total Records: {total_records}\n")
    f.write(f"Registered Students: {unique_students}\n")
    f.write(f"Present Today: {present_today}\n")
    f.write(f"Attendance Percentage: {attendance_percentage:.2f}%\n\n")

    f.write("Emotion Distribution\n")
    f.write("-" * 25 + "\n")

    for emotion, count in emotion_counts.items():
        f.write(f"{emotion}: {count}\n")

print("Reports Generated Successfully.")

print(f"\nSaved in:\n{REPORT_DIR}")
