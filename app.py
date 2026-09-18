import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#téléchargment model
MODEL_PATH = 'models/best_model.h5'
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: Model not found at {MODEL_PATH}")


# دالة Grad-CAM
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block16_concat"):
    conv_layer = None
    for layer in model.layers:
        if 'conv5' in layer.name or 'relu' in layer.name:
            conv_layer = layer.name
    if conv_layer:
        last_conv_layer_name = conv_layer

    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_gradcam(img_path, heatmap, output_path, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    cv2.imwrite(output_path, superimposed_img)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = "uploaded_mri.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # معالجة الصورة والتنبؤ
            try:
                image = Image.open(filepath).convert("RGB")
                img_resized = image.resize((224, 224))
                img_array_pred = np.array(img_resized) / 255.0
                img_array_pred = np.expand_dims(img_array_pred, axis=0)

                if model is not None:
                    prediction = float(model.predict(img_array_pred)[0][0])
                    is_tumor = prediction > 0.5
                    confidence = prediction * 100 if is_tumor else (1 - prediction) * 100

                    # توليد Grad-CAM
                    gradcam_filename = "gradcam_result.jpg"
                    gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
                    try:
                        heatmap = make_gradcam_heatmap(img_array_pred, model)
                        save_gradcam(filepath, heatmap, gradcam_path)
                    except Exception as e:
                        print(f"Grad-CAM error: {e}")
                        gradcam_filename = None

                    return render_template('index.html',
                                           prediction=is_tumor,
                                           confidence=round(confidence, 2),
                                           original_image=url_for('static', filename='uploads/' + filename),
                                           gradcam_image=url_for('static',
                                                                 filename='uploads/' + gradcam_filename) if gradcam_filename else None)
                else:
                    return "Error: AI Model is not loaded! Check terminal logs.", 500
            except Exception as e:
                return f"Error processing image: {e}", 500

    return render_template('index.html', prediction=None)


if __name__ == '__main__':
    app.run(debug=True, port=5000)