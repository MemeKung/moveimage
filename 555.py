import os
import shutil
import glob
import tensorflow as tf
import numpy as np
import sys
from PIL import Image, UnidentifiedImageError

sys.stdout.reconfigure(encoding='utf-8')

from tensorflow import keras
from tensorflow.keras import layers

# ตรวจสอบ GPU และตั้งค่า memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("พบ GPU จำนวน:", len(gpus), "และตั้งค่า memory growth เรียบร้อยแล้ว")
    except RuntimeError as e:
        print("เกิดข้อผิดพลาดในการตั้งค่า GPU:", e)
else:
    print("ไม่พบ GPU ระบบจะใช้ CPU ในการฝึกโมเดล")

# ---------------------------------------
# ✅ ลบไฟล์ที่ไม่ใช่ภาพ หรือเปิดไม่ได้ด้วย PIL
# ---------------------------------------
def remove_invalid_images(data_dir, valid_exts=[".jpg", ".jpeg", ".png", ".gif", ".bmp"]):
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            full_path = os.path.join(root, file)

            if ext not in valid_exts:
                print(f"[ลบ] ไฟล์นามสกุลไม่รองรับ: {full_path}")
                os.remove(full_path)
                continue

            try:
                with Image.open(full_path) as img:
                    img.verify()
            except Exception as e:
                print(f"[ลบ] ไฟล์เสียหรือเปิดไม่ได้: {full_path} ({e})")
                os.remove(full_path)

# ---------------------------------------
# ✅ ลบไฟล์ที่ TensorFlow decode ไม่ได้
# ---------------------------------------
def remove_unreadable_images(data_dir):
    print("[🔍] กำลังตรวจสอบว่า TensorFlow สามารถ decode ภาพได้หรือไม่...")
    unreadable_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                img = tf.io.read_file(filepath)
                _ = tf.io.decode_image(img)
            except Exception as e:
                unreadable_files.append(filepath)
                print(f"[ลบ] TensorFlow อ่านไม่ได้: {filepath} ({e})")
                os.remove(filepath)
    print(f"[✅] ตรวจเสร็จ ลบไป {len(unreadable_files)} ไฟล์")

# ---------------------------------------
# 1) ฟังก์ชันฝึกโมเดล
# ---------------------------------------
def train_model(train_dir, img_size=(224, 224), batch_size=32, epochs=5):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    print("Class names:", class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    with tf.device('/GPU:0'):
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(img_size[0], img_size[1], 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        inputs = keras.Input(shape=(img_size[0], img_size[1], 3))
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(len(class_names), activation='softmax')(x)
        model = keras.Model(inputs, outputs)

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    return model, class_names

# ---------------------------------------
# 2) ฟังก์ชันทำนายภาพเดี่ยว
# ---------------------------------------
def classify_image(model, class_names, img_path, img_size=(224, 224)):
    try:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        scores = predictions[0]

        max_index = np.argmax(scores)
        predicted_label = class_names[max_index]
        confidence = float(scores[max_index])

        return predicted_label, confidence
    except Exception as e:
        print(f"[ข้าม] ไม่สามารถทำนายไฟล์: {img_path} ({e})")
        return None, None

# ---------------------------------------
# 3) ย้ายไฟล์ไปตามคลาส
# ---------------------------------------
def move_image_to_folder(img_path, predicted_label, confidence, output_base_dir, threshold=0.8):
    if predicted_label is None:
        return None

    folder_name = predicted_label if confidence >= threshold else "Unknown"
    target_folder = os.path.join(output_base_dir, folder_name)
    os.makedirs(target_folder, exist_ok=True)

    basename = os.path.basename(img_path)
    target_path = os.path.join(target_folder, basename)
    shutil.move(img_path, target_path)

    print(f"Moved {img_path} -> {target_path}  (label={folder_name}, conf={confidence:.2f})")
    return (img_path, target_path, folder_name, confidence)

# ---------------------------------------
# 4) main function
# ---------------------------------------
def main():
    train_dir = "train_data"
    input_dir = "input_images"
    output_dir = "sorted_images"

    print("[🧼] ลบไฟล์ภาพที่ไม่รองรับ...")
    remove_invalid_images(train_dir)

    print("[🔎] ลบไฟล์ภาพที่ TensorFlow decode ไม่ได้...")
    remove_unreadable_images(train_dir)

    print("[🚀] เริ่มฝึกโมเดล...")
    model, class_names = train_model(
        train_dir=train_dir,
        img_size=(224, 224),
        batch_size=16,
        epochs=5
    )

    valid_exts = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp"]
    all_image_paths = []
    for ext in valid_exts:
        all_image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    threshold = 0.8
    move_log = []

    for img_path in all_image_paths:
        predicted_label, confidence = classify_image(
            model, class_names, img_path, img_size=(224, 224)
        )

        result = move_image_to_folder(
            img_path, predicted_label, confidence, output_base_dir=output_dir, threshold=threshold
        )
        if result:
            move_log.append(result)

    print("\nSummary of file moves:")
    for original, target, folder, conf in move_log:
        print(f"File '{original}' moved to folder '{folder}' at '{target}' (confidence: {conf:.2f})")

    print("\nDone!")

if __name__ == "__main__":
    main()
