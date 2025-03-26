import os
import shutil
import numpy as np
import tensorflow as tf
import cv2
import sys
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image, UnidentifiedImageError

sys.stdout.reconfigure(encoding='utf-8')

# 🔹 กำหนดโฟลเดอร์หลัก
TRAIN_DIR = "training_data"      
INPUT_DIR = "input_images"       
OUTPUT_DIR = "sorted_images"     
LOG_FILE = "file_movement_log.txt"  

SUPPORTED_FORMATS = [".webp", ".jpeg", ".jpg", ".png"]

# 🔹 โหลดข้อมูลจากโฟลเดอร์ตัวอย่าง
def load_training_data():
    labels = []
    images = []
    class_names = sorted(os.listdir(TRAIN_DIR))  

    for label_index, class_name in enumerate(class_names):
        class_path = os.path.join(TRAIN_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        for filename in os.listdir(class_path):
            if any(filename.lower().endswith(ext) for ext in SUPPORTED_FORMATS):
                img_path = os.path.join(class_path, filename)
                try:
                    with Image.open(img_path) as img:
                        img.verify()  
                    image = load_img(img_path, target_size=(64, 64))  
                    image = img_to_array(image) / 255.0  
                    images.append(image)
                    labels.append(label_index)
                except UnidentifiedImageError:
                    print(f"[❌] ไฟล์เสียหาย: {img_path}")
                except Exception as e:
                    print(f"[⚠️] ข้อผิดพลาดขณะโหลดรูป: {img_path} | Error: {e}")

    return np.array(images), to_categorical(labels, num_classes=len(class_names)), class_names

# 🔹 โหลดข้อมูลสำหรับการฝึก AI
X, y, class_names = load_training_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 สร้างโมเดล CNN
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(len(class_names), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 🔹 สร้างโฟลเดอร์ OUTPUT ถ้ายังไม่มี
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🔹 ฟังก์ชันบันทึก log
def log_message(message):
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(message + "\n")
    print(message)

# 🔹 ฟังก์ชันจัดเรียงภาพ
def classify_and_move_images():
    log_message("\n[INFO] เริ่มกระบวนการจัดเรียงไฟล์...\n")
    unmoved_files = []

    for filename in os.listdir(INPUT_DIR):
        if any(filename.lower().endswith(ext) for ext in SUPPORTED_FORMATS):
            img_path = os.path.join(INPUT_DIR, filename)

            try:
                with Image.open(img_path) as img:
                    img.verify()

                image = load_img(img_path, target_size=(64, 64))
                image = img_to_array(image) / 255.0
                image = np.expand_dims(image, axis=0)

                predictions = model.predict(image)
                predicted_class = np.argmax(predictions)
                predicted_label = class_names[predicted_class]

                # 🔹 ตรวจสอบว่ามีโฟลเดอร์ OUTPUT สำหรับหมวดหมู่ที่ทำนายหรือไม่
                output_class_path = os.path.join(OUTPUT_DIR, predicted_label)
                if not os.path.exists(output_class_path):
                    os.makedirs(output_class_path)  
                    log_message(f"[📂] สร้างโฟลเดอร์ใหม่: {predicted_label}")

                # 🔹 ย้ายไฟล์ไปยังโฟลเดอร์ที่ถูกต้อง
                dest_path = os.path.join(output_class_path, filename)
                shutil.move(img_path, dest_path)
                log_message(f"[✅] {filename} → {predicted_label}/")

            except UnidentifiedImageError:
                unmoved_files.append(filename)
                log_message(f"[❌] ไฟล์เสียหายหรือไม่ใช่รูปภาพ: {filename}")
            except Exception as e:
                unmoved_files.append(filename)
                log_message(f"[❌] ไม่สามารถย้ายไฟล์: {filename} | Error: {e}")

    # 🔹 ตรวจสอบว่ามีไฟล์ที่ไม่สามารถย้ายได้หรือไม่
    if unmoved_files:
        log_message("\n⚠️ ไฟล์ที่ไม่สามารถย้ายได้:")
        for file in unmoved_files:
            log_message(f"   - {file}")

# 🔹 เรียกใช้ฟังก์ชันจัดเรียงภาพ
classify_and_move_images()
