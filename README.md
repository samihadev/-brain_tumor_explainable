# -brain_tumor_explainable
# 🧠 Explainable AI for Brain Tumor Detection

An interactive web application built with **Flask** and **TensorFlow / Keras** for detecting brain tumors from MRI scans using a deep learning pipeline (**DenseNet-121**), integrated with **Grad-CAM** for visual explainability and transparent AI diagnostic decision-making.

---

## ✨ Features

* **Deep Learning Prediction:** Utilizes a pre-trained convolutional neural network to classify brain MRI scans (Tumor Detected vs. Normal).
* **Explainable AI (Grad-CAM):** Automatically generates heatmaps highlighting the exact regions of the brain scan that influenced the model's prediction.
* **Modern 3D-Styled Glassmorphism UI:** Custom-designed Flask interface featuring glowing neon aesthetics, smooth 3D floating panels, and animated typography.
* **Robust Diagnostic Pipeline:** Fast processing, automated image resizing, and error handling for safe execution.

---

## 🛠️ Project Structure

```text
brain_tumor_explainable/
│
├── app.py                      # Main Flask application file
├── models/
│   └── best_model.h5           # Pre-trained DenseNet-121 deep learning model
├── static/
│   └── uploads/                # Temporary storage for uploaded MRIs and heatmaps
└── templates/
    └── index.html              # Frontend user interface (3D-styled HTML/CSS)
