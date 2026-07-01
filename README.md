# 🛡️ Intelligent Document Classifier & ID Extractor

A lightweight, CPU-optimized desktop application built with Python and PyQt6 that automatically classifies official documents (**T.C. ID Cards, Driving Licences, and Passports**) and extracts verified **T.C. ID / Passport Numbers** from images.

This project is fully optimized to run on standard computers (e.g., 8 GB RAM) **without requiring a dedicated graphics card (GPU)**.

---

## ✨ Key Features

* **Smart Document Classification:** Analyzes text using a keyword-scoring mechanism to identify if the uploaded image is an ID card, driver's license, or passport.
* **No Frozen UI (Multi-Threading):** Heavy text recognition (OCR) processes run safely in the background using `QThread`. The app interface remains 100% fluid and never freezes.
* **Advanced Image Enhancement:** Automatically cleans up image noise, balances flash glares using OpenCV (CLAHE + Otsu Thresholding), and prepares the text for maximum OCR accuracy.
* **Algorithmic Validation:** Extracted 11-digit numbers are automatically checked against the official T.C. ID mathematical algorithm to filter out false positives.
* **Clean Architecture:** Built using the **Dependency Injection** pattern, making the code highly modular, readable, and easy to maintain.

---

## 🛠️ Built With

* **Python 3**
* **PyQt6** (Graphical User Interface)
* **OpenCV** (Image Preprocessing)
* **Tesseract OCR** (Text Recognition Engine)
* **NumPy**

---

## 🔧 Installation & Setup

### 1. Install Python Dependencies
Open your terminal/command prompt in the project folder and run:
```bash
pip install -r requirements.txt