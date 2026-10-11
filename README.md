# Handwritten Digit Recognition System Using Convolutional Neural Networks (CNN)

An end-to-end Machine Learning and Computer Vision application designed for classifying handwritten digits (0–9) using Deep Learning. Built with **TensorFlow / Keras**, **NumPy**, **Scikit-learn**, and deployed as an interactive web dashboard using **Streamlit**.

---

## 1. Abstract

Handwritten digit recognition is a foundational computer vision problem that bridges mathematical pattern recognition and deep learning. Due to the high degree of variability in human handwriting, rule-based systems struggle to achieve high precision. This project implements a **Convolutional Neural Network (CNN)** trained on the benchmark **MNIST dataset** (70,000 images). The system features an interactive web interface where users can draw digits on a virtual canvas, which are then preprocessed using aspect-ratio preservation and center-of-mass alignment before being classified by the trained model with class probability distributions.

---

## 2. Problem Statement

Automated recognition of handwritten numerals is essential in postal automation, bank check clearance, and document digitization. However, variations in stroke thickness, slant, size, and individual writing idiosyncrasies create significant visual ambiguity. The objective is to construct a deep learning system that autonomously extracts spatial feature hierarchies from raw pixel representations and generalizes reliably to unseen human handwriting.

---

## 3. Objectives

- **Dataset Ingestion**: Load, parse, normalize, and reshape 70,000 MNIST grayscale images (60,000 train/val and 10,000 test).
- **Deep Learning Model**: Design, train, regularize, and evaluate a CNN architecture adhering to academic standards.
- **Strict Test Separation**: Maintain an untouched test partition to eliminate data leakage and ensure unbiased generalization assessment.
- **Evaluation Suite**: Compute test accuracy, loss, precision, recall, F1-score, confusion matrix, and sample prediction grids.
- **Interactive Web App**: Build a modern Streamlit application with a canvas drawing tool, real-time prediction, probability breakdown, and preprocessing transparency.
- **Robustness**: Implement input validation, empty canvas detection, and graceful error handling.

---

## 4. Hardware and Software Requirements

### Hardware Requirements
- **Processor**: Intel Core i3 / AMD Ryzen 3 or higher (i5/Ryzen 5 recommended)
- **RAM**: Minimum 4 GB (8 GB recommended)
- **Disk Space**: At least 1 GB of free space for datasets, models, and dependencies
- **Display**: 1366 × 768 or higher resolution

### Software Requirements
- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: Version 3.10 – 3.12
- **Core Libraries**:
  - `tensorflow` (>= 2.15.0)
  - `numpy` (>= 1.24.0)
  - `scikit-learn` (>= 1.3.0)
  - `matplotlib` (>= 3.8.0)
  - `pillow` (>= 10.0.0)
  - `scipy` (>= 1.11.0)
  - `streamlit` (>= 1.30.0)
  - `streamlit-drawable-canvas` (0.9.3)
  - `plotly` (>= 5.18.0)

---

## 5. Dataset Description (MNIST)

The **MNIST (Modified National Institute of Standards and Technology)** dataset is the standard benchmark for handwritten digit classification:
- **Total Images**: 70,000 grayscale digit images (0 through 9)
- **Training Set**: 50,000 images
- **Validation Set**: 10,000 images (derived from training data for hyperparameter monitoring and early stopping)
- **Test Set**: 10,000 images (strictly untouched until final model evaluation)
- **Resolution**: 28 × 28 pixels (784 total input features per image)
- **Value Range**: Pixel intensity from 0 (black background) to 255 (white stroke), normalized to $[0.0, 1.0]$.

The project supports both direct IDX/ubyte parsing from the local dataset folder (`handwritten data set/`) and automatic fallback to `tf.keras.datasets.mnist`.

---

## 6. System Architecture and Workflow

```
[ User Input (Canvas / File) ]
              │
              ▼
[ Grayscale & Polarity Inversion ]
              │
              ▼
[ Bounding Box Crop (Active Pixels) ]
              │
              ▼
[ Aspect Ratio Resize (Max 20×20) ]
              │
              ▼
[ Center-of-Mass Alignment (28×28 Frame) ]
              │
              ▼
[ Normalization to [0.0, 1.0] -> Shape (1, 28, 28, 1) ]
              │
              ▼
[ Trained CNN Model (models/mnist_cnn.keras) ]
              │
              ▼
[ Softmax Probabilities & Predicted Class (0-9) ]
              │
              ▼
[ Interactive UI: Digit Badge, Confidence, Probability Chart ]
```

---

## 7. Convolutional Neural Network (CNN) Architecture

| Layer | Type | Specifications | Output Shape | Activation |
|---|---|---|---|---|
| **Input** | `Input` | Grayscale tensor input | `(28, 28, 1)` | — |
| **Conv 1** | `Conv2D` | 32 filters, 3×3 kernel, padding `'same'` | `(28, 28, 32)` | ReLU |
| **Pool 1** | `MaxPooling2D` | 2×2 pool size, stride 2 | `(14, 14, 32)` | — |
| **Conv 2** | `Conv2D` | 64 filters, 3×3 kernel, padding `'same'` | `(14, 14, 64)` | ReLU |
| **Pool 2** | `MaxPooling2D` | 2×2 pool size, stride 2 | `(7, 7, 64)` | — |
| **Flatten**| `Flatten` | Flattens 3D feature representation | `(3136,)` | — |
| **Dense**  | `Dense` | 128 fully connected hidden units | `(128,)` | ReLU |
| **Dropout**| `Dropout` | Regularization rate: 0.30 | `(128,)` | — |
| **Output** | `Dense` | 10 output class logits/probabilities | `(10,)` | Softmax |

- **Loss Function**: Sparse Categorical Cross-Entropy (`sparse_categorical_crossentropy`)
- **Optimizer**: Adam (`learning_rate=0.001`)
- **Regularization**: Dropout (0.30) and EarlyStopping on `val_loss` (patience=3, restoring best weights).

---

## 8. Project Structure

```
Handwritten Digit Recognition System/
├── app.py                      # Streamlit interactive web dashboard
├── train.py                    # Model training pipeline with device auto-targeting
├── evaluate.py                 # Evaluation pipeline on untouched test set
├── predict.py                  # Single-image prediction CLI and module
├── verify_device.py            # Hardware verification & GPU diagnostic suite
├── requirements.txt            # Project dependencies
├── README.md                   # Complete project documentation
├── handwritten data set/       # Local raw MNIST ubyte IDX dataset files
│   ├── train-images.idx3-ubyte
│   ├── train-labels.idx1-ubyte
│   ├── t10k-images.idx3-ubyte
│   └── t10k-labels.idx1-ubyte
├── models/
│   └── mnist_cnn.keras         # Saved trained Keras model artifact
├── results/
│   ├── evaluation_metrics.json # Structured test evaluation metrics
│   ├── classification_report.txt# Detailed precision/recall/f1 per digit
│   ├── confusion_matrix.png    # Heatmap of actual vs predicted classes
│   ├── training_accuracy.png   # Train vs validation accuracy per epoch
│   ├── training_loss.png       # Train vs validation loss per epoch
│   └── sample_predictions.png  # Test sample prediction grid
├── src/
│   ├── __init__.py
│   ├── device.py               # GPU auto-detection, memory growth, CPU fallback
│   ├── preprocessing.py        # Dataset parsing, scaling & centering
│   └── model.py                # CNN architecture builder & loader
└── tests/
    ├── test_device.py          # Unit tests for hardware configuration & fallback
    ├── test_preprocessing.py   # Unit tests for preprocessing & shapes
    └── test_prediction.py      # Unit tests for inference & probabilities
```

---

## 9. Hardware Acceleration & NVIDIA RTX 3050 (WSL2 Guide)

### Why WSL2 for TensorFlow GPU Acceleration?
Official TensorFlow packages for Windows ceased native GPU support starting with **TensorFlow 2.11**. To run modern TensorFlow (>= 2.15) with GPU acceleration on your **NVIDIA GeForce RTX 3050 (4 GB VRAM)**, TensorFlow officially partners with Microsoft and NVIDIA via **WSL2 (Windows Subsystem for Linux)**. The Windows NVIDIA display driver automatically provides the CUDA driver library (`/usr/lib/wsl/lib/libcuda.so.1`) directly to WSL2.

### Automatic Device Management & Dynamic Memory Growth
1. **Device Auto-Detection**: At launch, `src/device.py` queries `tf.config.list_physical_devices('GPU')`.
2. **GPU Memory Growth**: By default, TensorFlow attempts to pre-allocate almost all GPU VRAM. On an RTX 3050 with 4 GB VRAM, this project automatically enables **dynamic memory growth** (`tf.config.experimental.set_memory_growth(gpu, True)`), allocating only the VRAM needed as training progresses and preventing Out-Of-Memory (OOM) errors.
3. **Graceful CPU Fallback**: If no GPU is registered (such as when running on native Windows), the system automatically maps operations to `/device:CPU:0` with a clean diagnostic log. GPU acceleration is **never** falsely claimed.

### Step-by-Step Instructions to Run on GPU in WSL2

#### 1. Open WSL2 Ubuntu Terminal
From Windows PowerShell or Command Prompt, launch Ubuntu:
```bash
wsl -d Ubuntu
```

#### 2. Confirm NVIDIA Driver Passthrough
Verify that your RTX 3050 is visible inside WSL2:
```bash
nvidia-smi
```
*(You will see `NVIDIA GeForce RTX 3050 Laptop GPU` or Desktop GPU with Driver version and CUDA support).*

#### 3. Navigate to the Project Workspace
From within WSL2, navigate to the Windows project path:
```bash
cd "/mnt/d/projects/Handwritten Digit Recognition System"
```

#### 4. Set Up Python & Install Dependencies
Ensure Python virtual environment tools and dependencies are installed:
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv
python3 -m venv venv_wsl
source venv_wsl/bin/activate
pip install --upgrade pip
pip install "tensorflow[and-cuda]" -r requirements.txt
```

#### 5. Verify GPU Hardware Acceleration
Run the device verification script:
```bash
python verify_device.py
```
**Expected Output in WSL2:**
```text
======================================================================
 HANDWRITTEN DIGIT RECOGNITION - DEVICE VERIFICATION SUITE
======================================================================
 Operating System  : Linux ...
 Environment Type  : WSL2 Linux Environment
 Python Version    : 3.12.x
[NVIDIA Driver Check]
 -> nvidia-smi is available: NVIDIA GeForce RTX 3050, 4096 MiB
[TensorFlow Device Configuration]
 TensorFlow Version  : 2.x
 Detected GPU Devices: 1 (/physical_device:GPU:0)
 Memory Growth       : Enabled
 Execution Target    : GPU (Hardware Accelerated)
[Test Benchmark] Running test matrix multiplication on /device:GPU:0...
 [PASS] TensorFlow GPU acceleration is CONFIRMED and ACTIVE on /physical_device:GPU:0!
======================================================================
```

#### 6. Train on GPU in WSL2
```bash
python train.py --epochs 10 --batch-size 128
```

#### 7. Launch Streamlit Web App from WSL2
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your Windows browser. The sidebar will display: `⚡ GPU: /physical_device:GPU:0`.

---

## 10. Execution Commands

### Step 1: Verify Hardware Device Target
Checks your environment, detects available GPU/CPU devices, sets memory growth, and performs a test benchmark:
```bash
python verify_device.py
```

### Step 2: Train the CNN Model
Trains the CNN model for up to 10 epochs using 50,000 training images and 10,000 validation images. Maps operations to GPU if present, otherwise executes cleanly on CPU:
```bash
python train.py --epochs 10 --batch-size 128
```

### Step 3: Evaluate the Model
Evaluates the model on the 10,000 untouched test samples. Computes accuracy, loss, precision, recall, F1, and produces the confusion matrix:
```bash
python evaluate.py
```

### Step 4: Run Automated Unit Tests
Verifies hardware detection, preprocessing, tensor normalization, empty canvas exceptions, and prediction output shapes:
```bash
python -m unittest discover tests
```

### Step 5: Launch the Streamlit Web Application
Starts the interactive user interface on `http://localhost:8501`:
```bash
streamlit run app.py
```

### Step 6: Command-Line Single Image Prediction (Optional)
Run inference on an individual image file:
```bash
python predict.py --image "path/to/digit.png"
```


---

## 11. Performance Metrics and Results

Evaluation metrics generated on the untouched MNIST test dataset (10,000 images):

| Metric | Score |
|---|---|
| **Test Accuracy** | ~99.1% |
| **Test Loss** | ~0.028 |
| **Macro Precision** | ~99.1% |
| **Macro Recall** | ~99.1% |
| **Macro F1-Score** | ~99.1% |
| **Total Test Samples** | 10,000 |

### Why the Test Set Must Remain Separate
In rigorous machine learning methodology, the test set must remain completely isolated and unseen throughout model development. Network weights are updated via backpropagation on the training set, while the validation set tunes hyperparameters and guides early stopping. If test samples were used during training or model selection, **data leakage** would occur, resulting in an overly optimistic evaluation that fails to generalize in real-world deployment.

---

## 12. Limitations and Future Scope

### Limitations
- The system classifies isolated single digits (0–9) and does not segment continuous multi-digit sequences (e.g., telephone numbers or zip codes).
- Highly tilted or deformed strokes may experience reduced confidence if outside the distribution of the training set.

### Future Scope
- **Multi-Digit Segmentation**: Integrate contour detection or bounding-box clustering to transcribe multi-digit numbers.
- **Data Augmentation**: Train with random affine transformations (rotations of $\pm 15^\circ$, shearing, scaling) to increase robustness against extreme handwriting slants.
- **Mobile Stylus Support**: Add pressure-sensitive stroke thickness rendering.

---

## 13. Conclusion

This project successfully demonstrates a full-stack, modular, and academically rigorous machine learning mini project for handwritten digit recognition. By combining a Convolutional Neural Network with careful stroke preprocessing (bounding-box normalization and center-of-mass alignment), the system achieves over 99% test accuracy on the benchmark MNIST dataset while providing a responsive, intuitive web application for real-time interaction.
