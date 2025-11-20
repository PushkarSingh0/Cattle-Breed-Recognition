# 🐄 Cattle Breed Recognition System (SIH 2025)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B)
![Status](https://img.shields.io/badge/Status-Prototype-success)

**An AI-powered computer vision system designed to identify indigenous Indian cattle and buffalo breeds.** Developed as part of the **Smart India Hackathon (SIH) 2025** (Problem Statement ID: SIH25099), this project utilizes advanced Deep Learning architectures to assist farmers, veterinarians, and insurance agencies in accurate breed identification.

---

## 🚀 Key Features

### 1. 🛡️ The "Gatekeeper" System
Unlike standard classifiers that force a prediction on any image, our system employs a **Gatekeeper Model** (Binary Classifier).
- **Function:** It first checks if the uploaded image actually contains a cow or buffalo.
- **Result:** If the image is irrelevant (e.g., a car, a person, or a generic landscape), the system rejects it immediately, preventing false classifications.

### 2. 🧬 Breed Classification
- **Architecture:** Fine-tuned **EfficientNet-B3** (Transfer Learning).
- **Capabilities:** Classification of multiple Indian cattle and buffalo breeds with high confidence.
- **Weights:** Pre-trained models stored locally for fast inference.

### 3. 💻 Interactive Web UI
- Built with **Streamlit** for a user-friendly, responsive interface.
- Real-time inference speed.

---

## 📂 Project Structure

```bash
Cattle-Breed-Recognition/
├── 📁 app/                  # Application Layer
│   ├── server.py            # Backend inference server (loads models)
│   └── streamlit_app.py     # Frontend UI (Streamlit)
├── 📁 models/               # Trained Model Weights
│   ├── best_breed_classifier_b3.pt
│   ├── gatekeeper_model.pt
│   └── breed_class_names.json
├── 📁 src/                  # Source Code for Training
│   ├── train.py             # Main training loop
│   ├── model.py             # Model architecture definitions
│   ├── evaluate.py          # Performance metrics
│   └── prepare_data.py      # Data preprocessing pipeline
├── requirements.txt         # Project dependencies
└── README.md                # Project Documentation
