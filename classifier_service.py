import pytesseract
import re
import numpy as np

# Tesseract'ın Windows işletim sistemindeki motor yolu
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class ClassifierService:
    # DEPENDENCY INJECTION: Alt katman motoru constructor üzerinden içeri enjekte ediliyor
    def __init__(self, core_utils_instance):
        self.utils = core_utils_instance 

    def run_pipeline(self, img_bgr: np.ndarray) -> dict:
        """Belge Sınıflandırma ve Sadece T.C./Pasaport No Kazıma Akışı"""
        
        # CPU dostu görüntü ön işleme adımını tetikliyoruz
        ocr_ready_img = self.utils.preprocess_image(img_bgr)
        
        # Tesseract geniş alan taraması (--psm 11 formatı dağınık metinlerde kelime yakalamak için idealdir)
        raw_text = pytesseract.image_to_string(ocr_ready_img, lang='tur+eng', config='--oem 3 --psm 11')
        upper_text = raw_text.upper()
        
        # Semantik İmza Kelime Havuzları (Belgeyi tanımaya yarayan anahtar kelimeler)
        signatures = {
            "T.C. KİMLİK KARTI": ["KİMLİK", "IDENTITY", "TURKEY", "SOYADI", "ADI", "TCK", "KARTI"],
            "SÜRÜCÜ BELGESİ (EHLİYET)": ["SÜRÜCÜ", "DRIVING", "LICENCE", "EHLİYET", "KATEGORİ", "BELGESİ"],
            "PASAPORT (PASSPORT)": ["PASAPORT", "PASSPORT", "P<TUR", "NATIONALITY", "PASAPORTU"]
        }
        
        # Skorlama Mekanizması
        scores = {doc_type: 0 for doc_type in signatures}
        detected_kws = []
        for doc_type, keywords in signatures.items():
            for kw in keywords:
                if kw in upper_text:
                    scores[doc_type] += 1
                    if kw not in detected_kws: 
                        detected_kws.append(kw)
                    
        # Güvenli Skorlama Kontrolü
        if sum(scores.values()) == 0:
            doc_class = "UNKNOWN / OKUNAMADI"
            confidence = 0.0
        else:
            best_match = max(scores, key=scores.get)
            doc_class = best_match
            confidence = (scores[best_match] / len(signatures[best_match])) * 100

        # Sadece Kritik Numara Çıkarımı Kontrolü
        extracted_data = "Kritik veri saptanamadı"
        
        if doc_class in ["T.C. KİMLİK KARTI", "SÜRÜCÜ BELGESİ (EHLİYET)"]:
            # Başında 0 olmayan 11 haneli sayıları ara
            potential_tcs = re.findall(r'\b[1-9]\d{10}\b', upper_text)
            for tc in potential_tcs:
                # Çekirdek algoritma fonksiyonuna T.C. No'yu doğrulatıyoruz
                if self.utils.validate_tc_algorithm(tc):
                    extracted_data = f"DOĞRULANMIŞ T.C. NO: {tc}"
                    break # İlk doğru T.C. numarasını bulduğumuzda döngüden çıkıyoruz
                    
        elif doc_class == "PASAPORT (PASSPORT)":
            # Pasaport numarası formatı (1 harf ve 7-8 rakam) arama
            passport_matches = re.findall(r'\b[A-Z0-9]{1}[0-9]{7,8}\b', upper_text)
            if passport_matches: 
                extracted_data = f"PASAPORT NO: {passport_matches[0]}"

        return {
            "status": "SUCCESS" if doc_class != "UNKNOWN / OKUNAMADI" else "FAILED",
            "document_type": doc_class,
            "confidence": round(confidence, 2),
            "extracted_data": extracted_data,
            "detected_keywords": detected_kws,
            "raw_text": raw_text
        }