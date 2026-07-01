import cv2
import numpy as np
import re

class CoreUtils:
    def __init__(self):
        """Configuration hub for potential future expansions (e.g., specific Tesseract paths)"""
        pass

    def validate_tc_algorithm(self, tc_str: str) -> bool:
        """Validates Turkish ID Number based on official 11-digit mathematical algorithm rules"""
        if len(tc_str) != 11 or not tc_str.isdigit() or tc_str[0] == '0':
            return False
        
        digits = [int(d) for d in tc_str]
        
        # Calculate odd and even position sums (Slicing technique)
        odd_sum = sum(digits[0:9:2])   # 1st, 3rd, 5th, 7th, 9th digits
        even_sum = sum(digits[1:8:2])  # 2nd, 4th, 6th, 8th digits
        
        # 1st Rule Validation for the 10th digit
        if ((odd_sum * 7) - even_sum) % 10 != digits[9]:
            return False
            
        # 2nd Rule Validation for the 11th digit
        if sum(digits[0:10]) % 10 != digits[10]:
            return False
            
        return True

    def preprocess_image(self, img_bgr: np.ndarray) -> np.ndarray:
        """Advanced CPU-friendly image enhancement and noise reduction pipeline"""
        # Convert to Grayscale to drop channels
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization) to eliminate flash glares
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        contrast = clahe.apply(gray)
        
        # Maintain aspect ratio while locking width to an optimal 1600px for CPU-OCR performance
        h, w = contrast.shape[:2]
        target_w = 1600
        target_h = int(h * (target_w / w))
        resized = cv2.resize(contrast, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        
        # Apply slight blur to suppress background security pattern noises
        blurred = cv2.GaussianBlur(resized, (3, 3), 0)
        
        # Otsu's Binarization to force clean black & white matrix
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh