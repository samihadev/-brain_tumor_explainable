import os
import yaml
import tensorflow as tf
from data import get_data_generators
from model import build_model

# تحديد المسار المطلق لملف الإعدادات
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(BASE_DIR, "config.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)


def train():
    img_size = tuple(config['data']['img_size'])
    batch_size = config['data']['batch_size']
    epochs = config['training']['epochs']
    learning_rate = config['training']['learning_rate']
    loss_fn = config['training']['loss']

    # مسار البيانات المحلي
    data_dir = os.path.join(BASE_DIR, "data/raw/Brain_Tumor_Dataset")

    train_gen, val_gen = get_data_generators(data_dir, img_size, batch_size)

    model, base_model = build_model(model_name='densenet121')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )

    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    model_checkpoint_path = os.path.join(BASE_DIR, "models/best_model.h5")

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_checkpoint_path, save_best_only=True, monitor='val_loss')
    ]

    print("بدء تدريب النموذج...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )
    print("تم التدريب وحفظ أفضل نموذج بنجاح!")


if __name__ == "__main__":
    train()