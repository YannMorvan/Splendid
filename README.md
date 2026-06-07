# 🎭 Face Mask Detection - Splendid

A real-time face mask detection system using PyTorch and MobileNetV3, with support for inference on individual images and webcam detection.

## 📋 Table of Contents

- [Installation](#installation)
- [Usage - Demo](#usage---demo)
- [Usage - Web Application](#usage---web-application)
- [Dataset](#dataset)
- [Project Structure](#project-structure)

---

## 🚀 Installation

### 1. **Prerequisites**

- Python 3.8+
- pip or conda
- GPU CUDA (optional but recommended for performance)

### 2. **Clone or download the project**

```bash
cd Splendid
```

### 3. **Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 4. **Install dependencies**

```bash
pip install -r requirements.txt
```

### 5. **Download the Kaggle dataset** (optional if you want to retrain)

The pre-trained model is already included in the `models/best_model.pt` folder.

---

## 📸 Usage - Demo

### Test on a single image

Use `demo.py` to test the model on a single image.

```bash
python demo.py path/to/image.jpg
```

**Available options:**

```bash
# With the default model (models/best_model.pt)
python demo.py path/to/image.jpg

# With a custom model
python demo.py path/to/image.jpg --model models/best_model.pt
```

**Example:**

```bash
python demo.py data/dataset/with_mask/example.jpg
```

**Expected output:**

```
Prediction: with_mask
Confidence: 0.95
```

---

## 🌐 Usage - Web Application

### Launch the Flask server

```bash
python app.py
```

**Available options:**

```bash
# With default parameters
python app.py

# With a custom port
python app.py --port 8080

# With a custom model
python app.py --model models/best_model.pt --port 5000
```

**Once launched:**

1. Open your browser at `http://localhost:5000`
2. Allow access to your webcam
3. Real-time detection works directly in the browser

---

## 📦 Dataset

The dataset can be downloaded from Kaggle:

**Option 1: Face Mask Dataset (recommended)**

- **Link:** https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
- **Description:** Complete dataset with and without masks

**Option 2: Face Mask Detection Dataset**

- **Link:** https://www.kaggle.com/datasets/andrewmvd/face-mask-detection

### Dataset installation

1. Download the dataset from Kaggle
2. Extract files into the `data/dataset/` folder:
    ```
    data/
    └── dataset/
        ├── with_mask/
        │   └── [images with mask]
        └── without_mask/
            └── [images without mask]
    ```

**Note:** A pre-trained model is provided in `models/best_model.pt`. You don't need to retrain to use the application!

---

## 📁 Project Structure

```
Splendid/
├── app.py                          # Flask server for webcam
├── demo.py                         # Image inference
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── models/
│   └── best_model.pt               # Pre-trained model
├── data/
│   └── dataset/
│       ├── with_mask/              # Images with mask
│       └── without_mask/           # Images without mask
├── src/
│   ├── __init__.py
│   ├── model.py                    # Model architecture
│   ├── dataset.py                  # Data loading
│   ├── train.py                    # Training script
│   ├── evaluate.py                 # Model evaluation
│   └── focal_loss.py               # Custom loss function
└── templates/
    └── index.html                  # Web interface
```

---

## 🛠️ Technology Stack

- **Framework:** PyTorch
- **Architecture:** MobileNetV3 Small
- **Web Interface:** Flask + HTML5/CSS3/JavaScript
- **Preprocessing:** Torchvision transforms
- **Normalization:** ImageNet (mean, std)

---

## 📝 Quick Commands

```bash
# Complete setup
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Quick test on an image
python demo.py data/dataset/with_mask/sample.jpg

# Launch web app
python app.py

# With GPU CUDA
CUDA_VISIBLE_DEVICES=0 python app.py
```

---

## 📄 License

Educational Project - Splendid

---

**For any questions or issues, check the source files or train the model from `src/train.py`!**
