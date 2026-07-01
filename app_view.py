import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QProgressBar, QTextEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage

# Import architectural layer dependencies
from core_utils import CoreUtils
from classifier_service import ClassifierService

# =========================================================================
# 🔥 ASYNCHRONOUS BACKGROUND THREAD (OcrWorker)
# =========================================================================
class OcrWorker(QThread):
    # Communication line to emit the final analytical report back to the UI Thread
    finished = pyqtSignal(dict)

    def __init__(self, service_instance, opencv_img):
        super().__init__()
        self.service = service_instance
        self.img = opencv_img

    def run(self):
        """Asynchronous execution logic running isolated from the Main Thread to prevent UI lag"""
        report = self.service.run_pipeline(self.img)
        self.finished.emit(report) 


class DocumentClassifierApp(QMainWindow):
    def __init__(self, service_instance):
        super().__init__()
        self.service = service_instance # Injected Business Logic Service
        self.opencv_img = None
        self.worker = None 
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("🛡️ Enterprise Document Classifier (PyQt6 + DI)")
        self.setGeometry(100, 100, 1000, 700)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        title_label = QLabel("🛡️ Intelligent Document Classifier & Extractor")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 18pt; font-weight: bold; color: #2c3e50; margin: 10px;")
        main_layout.addWidget(title_label)

        content_layout = QHBoxLayout()
        
        # Left Panel: Image View Area
        self.image_label = QLabel("Upload Document Image...\n(Supported: JPG, JPEG, PNG)")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #bdc3c7; background-color: #f8f9fa; min-width: 450px;")
        content_layout.addWidget(self.image_label)

        # Right Panel: Analytics & Reporting Logs
        report_layout = QVBoxLayout()
        
        self.result_label = QLabel("📊 Analysis Status: Awaiting Input...")
        self.result_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #2980b9;")
        report_layout.addWidget(self.result_label)

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        report_layout.addWidget(self.confidence_bar)

        self.data_label = QLabel("📋 Extracted Critical Data: -")
        self.data_label.setStyleSheet("font-size: 12pt; color: #27ae60; font-weight: bold; margin-top: 10px;")
        report_layout.addWidget(self.data_label)

        report_layout.addWidget(QLabel("🔍 Raw OCR Text Engine Output:"))
        self.raw_text_box = QTextEdit()
        self.raw_text_box.setReadOnly(True)
        report_layout.addWidget(self.raw_text_box)

        content_layout.addLayout(report_layout)
        main_layout.addLayout(content_layout)

        # Action Controllers (Buttons)
        button_layout = QHBoxLayout()
        
        self.btn_upload = QPushButton("📁 Load Image")
        self.btn_upload.setStyleSheet("padding: 10px; font-size: 11pt; background-color: #34495e; color: white;")
        self.btn_upload.clicked.connect(self.open_image_dialog)
        button_layout.addWidget(self.btn_upload)

        self.btn_process = QPushButton("🚀 Filter & Process Document")
        self.btn_process.setEnabled(False)
        self.btn_process.setStyleSheet("padding: 10px; font-size: 11pt; background-color: #2ecc71; color: white;")
        self.btn_process.clicked.connect(self.start_async_process) 
        button_layout.addWidget(self.btn_process)

        main_layout.addLayout(button_layout)

    def open_image_dialog(self):
        """Streams and optimizes complex image encodings preventing operating system path blockages"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Document Image", "", "Images (*.jpg *.jpeg *.png)")
        if file_path:
            # Safe binary stream reading protecting UI against Turkish/Special character file paths
            with open(file_path, "rb") as stream:
                bytes_data = bytearray(stream.read())
            numpy_array = np.asarray(bytes_data, dtype=np.uint8)
            self.opencv_img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
            
            if self.opencv_img is None:
                self.result_label.setText("❌ Error: OpenCV failed to decode the target image!")
                return

            # Display optimization: Rescale matrix to handle memory consumption gracefully
            h, w = self.opencv_img.shape[:2]
            if w > 800:
                new_w = 800
                new_h = int(h * (800 / w))
                display_img = cv2.resize(self.opencv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                display_img = self.opencv_img.copy()

            # Swap BGR matrix to PyQt compatible RGB standard
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            bytes_per_line = 3 * display_img.shape[1]
            q_img = QImage(display_img.data, display_img.shape[1], display_img.shape[0], bytes_per_line, QImage.Format.Format_RGB888)
            
            # Draw image smoothly onto the View panel
            pixmap = QPixmap.fromImage(q_img)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.width(), 
                self.image_label.height(), 
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            
            self.btn_process.setEnabled(True)
            self.result_label.setText("📊 Analysis Status: Image loaded, ready to fire.")

    def start_async_process(self):
        """Spawns the background worker protecting the UI against freezing during active processing"""
        if self.opencv_img is not None:
            # Temporary safety block lock on controllers
            self.btn_process.setEnabled(False)
            self.btn_upload.setEnabled(False)
            self.result_label.setText("🔍 Tesseract OCR Active. Running Multi-Thread Analysis...")
            
            # Deploy asynchronous worker
            self.worker = OcrWorker(self.service, self.opencv_img)
            self.worker.finished.connect(self.on_process_finished)
            self.worker.start()

    def on_process_finished(self, report):
        """Triggered automatically when background thread finishes data extraction"""
        # Unlock controllers
        self.btn_process.setEnabled(True)
        self.btn_upload.setEnabled(True)
        
        if report["status"] == "SUCCESS":
            self.result_label.setText(f"🎉 DETECTED DOCUMENT: {report['document_type']}")
            self.confidence_bar.setValue(int(report["confidence"]))
            self.data_label.setText(f"📋 {report['extracted_data']}")
        else:
            self.result_label.setText("❌ Document Classification Failed!")
            self.confidence_bar.setValue(0)
            self.data_label.setText("📋 Extracted Data: No critical data detected")

        self.raw_text_box.setText(report["raw_text"])

# =========================================================================
# ⚙️ CENTRAL ARCHITECTURAL COMPOSITION ROOT (Bağımlılık Enjeksiyon Merkezi)
# =========================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Core engine initialization
    core_utils_instance = CoreUtils()
    
    # Injection pipeline strategy
    classifier_service_instance = ClassifierService(core_utils_instance=core_utils_instance)
    
    # Deliver ready infrastructure directly to the User View layer
    window = DocumentClassifierApp(service_instance=classifier_service_instance)
    window.show()
    
    sys.exit(app.exec())