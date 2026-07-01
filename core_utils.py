# core_utils.py
import cv2
import numpy as np
import re

class CoreUtils:
    def __init__(self):
        # Gelecekte buraya özel konfigürasyonlar (Örn: Tesseract dil paketleri yolları) eklenebilir.
        pass

    def validate_tc_algorithm(self, tc_str: str) -> bool:
        """T.C. Kimlik No resmi 11 haneli matematiksel algoritma kontrolü"""
        if len(tc_str) != 11 or not tc_str.isdigit() or tc_str[0] == '0':
            return False
        
        digits = [int(d) for d in tc_str]
        
        odd_sum = sum(digits[0:9:2])
        even_sum = sum(digits[1:8:2])
        if ((odd_sum * 7) - even_sum) % 10 != digits[9]:
            return False
            
        if sum(digits[0:10]) % 10 != digits[10]:
            return False
            
        return True

    def preprocess_image(self, img_bgr: np.ndarray) -> np.ndarray:
        """Gelişmiş Görüntü İyileştirme ve Keskinleştirme Filtresi"""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # CLAHE (Lokal kontrast dengeleme ile parlamaları yok etme)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        contrast = clahe.apply(gray)
        
        # Yeni satır (CPU dostu, ışık hızında OCR):
        h, w = contrast.shape[:2]
        target_w = 1600
        target_h = int(h * (target_w / w))
        resized = cv2.resize(contrast, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        blurred = cv2.GaussianBlur(resized, (3, 3), 0)
        # Otsu Binarization (Saf siyah-beyaz matrise zorlama)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh