import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_data_generators(data_dir="data/raw/Brain_Tumor_Dataset", img_size=(224, 224), batch_size=32):
    """
    يبحث عن صور الأورام في كامل مسارات المشروع، ينظمها، ويُنشئ مولدات البيانات فوراً.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dataset_path = os.path.join(base_dir, data_dir)

    tumor_dir = os.path.join(target_dataset_path, "tumor")
    notumor_dir = os.path.join(target_dataset_path, "notumor")

    # التحقق مما إذا كانت الصور منظمة وجاهزة وموجودة بالفعل
    if os.path.exists(tumor_dir) and os.path.exists(notumor_dir) and len(os.listdir(tumor_dir)) > 0:
        print(f"تم العثور على البيانات الجاهزة في: {target_dataset_path}")
    else:
        print("جاري البحث الشامل عن صور المشروع وتنظيمها تلقائياً...")
        os.makedirs(tumor_dir, exist_ok=True)
        os.makedirs(notumor_dir, exist_ok=True)

        found_images = False
        # recherche 3la les photos
        for root, dirs, files in os.walk(base_dir):
            # نتجاوز مجلدات النظام والبيئة الافتراضية لسرعة البحث
            if any(p in root for p in ['.venv', '.git', '__pycache__', 'models']):
                continue

            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    src_file = os.path.join(root, file)
                    parent_folder = os.path.basename(root).lower()

                    # classification des photos ida nrml wla nn
                    if 'notumor' in parent_folder or 'no_tumor' in parent_folder:
                        dest_file = os.path.join(notumor_dir, file)
                    elif any(w in parent_folder for w in ['glioma', 'meningioma', 'pituitary', 'tumor']):
                        dest_file = os.path.join(tumor_dir, file)
                    else:
                        continue

                    if not os.path.exists(dest_file):
                        shutil.copy(src_file, dest_file)
                    found_images = True

        if not found_images or len(os.listdir(tumor_dir)) == 0:
            raise FileNotFoundError(
                "لم يتم العثور على صور أورام الدماغ في المشروع. "
                "يرجى التأكد من وجود مجلد يحتوي على الصور في مسار المشروع."
            )
        print("تم العثور على الصور وتنظيمها بنجاح تام!")

    print("جاري إنشاء مولد بيانات التدريب...")
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='reflect',
        validation_split=0.3
    )

    val_test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.3
    )

    train_generator = train_datagen.flow_from_directory(
        target_dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        seed=42
    )

    print("جاري إنشاء مولد بيانات التحقق والاختبار...")
    val_generator = val_test_datagen.flow_from_directory(
        target_dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        seed=42
    )

    return train_generator, val_generator