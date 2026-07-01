import pytesseract
import re
import numpy as np

# Default Tesseract installation path for Windows environment
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class ClassifierService:
    # 💉 DEPENDENCY INJECTION: CoreUtils instance injected via constructor
    def __init__(self, core_utils_instance):
        self.utils = core_utils_instance 

    def run_pipeline(self, img_bgr: np.ndarray) -> dict:
        """Executes full document classification and specific metadata extraction pipeline"""
        
        # Run CPU-optimized preprocessing filters
        ocr_ready_img = self.utils.preprocess_image(img_bgr)
        
        # Wide-area sparse layout parsing configuration (--psm 11)
        raw_text = pytesseract.image_to_string(ocr_ready_img, lang='tur+eng', config='--oem 3 --psm 11')
        upper_text = raw_text.upper()
        
        # Semantic Document Signature Keyword Pools
        signatures = {
            "TURKISH ID CARD": ["KİMLİK", "IDENTITY", "TURKEY", "SOYADI", "ADI", "TCK", "KARTI"],
            "DRIVING LICENCE": ["SÜRÜCÜ", "DRIVING", "LICENCE", "EHLİYET", "KATEGORİ", "BELGESİ"],
            "PASSPORT": ["PASAPORT", "PASSPORT", "P<TUR", "NATIONALITY", "PASAPORTU"]
        }
        
        # Scoring Engine Initialization
        scores = {doc_type: 0 for doc_type in signatures}
        detected_kws = []
        
        for doc_type, keywords in signatures.items():
            for kw in keywords:
                if kw in upper_text:
                    scores[doc_type] += 1
                    if kw not in detected_kws: 
                        detected_kws.append(kw)
                    
        # Safe Evaluation Gate
        if sum(scores.values()) == 0:
            doc_class = "UNKNOWN / UNREADABLE"
            confidence = 0.0
        else:
            best_match = max(scores, key=scores.get)
            doc_class = best_match
            confidence = (scores[best_match] / len(signatures[best_match])) * 100

        # Targeted Metadata Extraction Gate
        extracted_data = "No critical data detected"
        
        if doc_class in ["TURKISH ID CARD", "DRIVING LICENCE"]:
            # Match 11-digit numbers not starting with zero using Regex boundaries
            potential_tcs = re.findall(r'\b[1-9]\d{10}\b', upper_text)
            for tc in potential_tcs:
                # Ask injected dependency to execute algorithmic check
                if self.utils.validate_tc_algorithm(tc):
                    extracted_data = f"VERIFIED T.C. NO: {tc}"
                    break 
                    
        elif doc_class == "PASSPORT":
            # Match standard passport format (1 alpha character followed by 7-8 digits)
            passport_matches = re.findall(r'\b[A-Z0-9]{1}[0-9]{7,8}\b', upper_text)
            if passport_matches: 
                extracted_data = f"PASSPORT NO: {passport_matches[0]}"

        return {
            "status": "SUCCESS" if doc_class != "UNKNOWN / UNREADABLE" else "FAILED",
            "document_type": doc_class,
            "confidence": round(confidence, 2),
            "extracted_data": extracted_data,
            "detected_keywords": detected_kws,
            "raw_text": raw_text
        }