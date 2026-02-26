"""
Feature Extraction Module for Fake Academic Certificate Detection
Extracts structural, logo, seal, signature, and text features from certificate images
"""

import cv2
import numpy as np
import pytesseract
import re
from typing import Dict, Any, Tuple, List, Optional
import logging
from pathlib import Path

# Configure pytesseract path (Windows users - uncomment and set your path)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CertificateFeatureExtractor:
    """
    Extract all numeric features from certificate images for fake detection
    """
    
    def __init__(self, 
                 logo_template_path: Optional[str] = None,
                 seal_template_path: Optional[str] = None,
                 signature_template_path: Optional[str] = None):
        """
        Initialize feature extractor with optional templates
        
        Args:
            logo_template_path: Path to genuine logo template image
            seal_template_path: Path to genuine seal template image
            signature_template_path: Path to genuine signature template image
        """
        self.logo_template = None
        self.seal_template = None
        self.signature_template = None
        
        # Load templates if provided
        if logo_template_path and Path(logo_template_path).exists():
            self.logo_template = cv2.imread(logo_template_path, cv2.IMREAD_GRAYSCALE)
            logger.info(f"Logo template loaded: {logo_template_path}")
            
        if seal_template_path and Path(seal_template_path).exists():
            self.seal_template = cv2.imread(seal_template_path, cv2.IMREAD_GRAYSCALE)
            logger.info(f"Seal template loaded: {seal_template_path}")
            
        if signature_template_path and Path(signature_template_path).exists():
            self.signature_template = cv2.imread(signature_template_path, cv2.IMREAD_GRAYSCALE)
            logger.info(f"Signature template loaded: {signature_template_path}")
    
    def extract_all_features(self, image_path: str) -> Dict[str, Any]:
        """
        Main function: Extract all features from certificate image
        
        Args:
            image_path: Path to certificate image
            
        Returns:
            Dictionary containing all numeric features
        """
        features = {}
        
        try:
            # Load image
            logger.info(f"Processing: {image_path}")
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Cannot read image: {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Extract all feature categories
            features.update(self._extract_structural_features(gray, img))
            features.update(self._extract_logo_features(gray))
            features.update(self._extract_seal_features(gray))
            features.update(self._extract_signature_features(gray))
            features.update(self._extract_text_features(gray))
            features.update(self._extract_format_validation_features(gray))
            
            # Add image metadata
            features['image_height'] = gray.shape[0]
            features['image_width'] = gray.shape[1]
            features['aspect_ratio'] = gray.shape[1] / gray.shape[0]
            
            logger.info(f"✅ Extracted {len(features)} features")
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            # Return empty features with error flag
            features = {'error_flag': 1, 'error_message': str(e)}
        
        return features
    
    # ============================================================
    # 1. STRUCTURAL FEATURES
    # ============================================================
    
    def _extract_structural_features(self, gray: np.ndarray, img: np.ndarray) -> Dict[str, float]:
        """Extract structural/image quality features"""
        features = {}
        
        try:
            # Edge count using Canny
            edges = cv2.Canny(gray, 50, 150)
            features['edge_count'] = int(np.sum(edges > 0))
            features['edge_density'] = features['edge_count'] / (gray.shape[0] * gray.shape[1])
            
            # Contour count
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            features['contour_count'] = len(contours)
            
            # Image sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            features['image_sharpness'] = float(laplacian.var())
            
            # Noise level (pixel intensity std deviation)
            features['noise_level'] = float(np.std(gray))
            
            # Blur detection (alternative measure)
            features['blur_score'] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            
            # Brightness and contrast
            features['brightness'] = float(np.mean(gray))
            features['contrast'] = float(np.std(gray))
            
            # Histogram features
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            features['hist_peak'] = float(np.argmax(hist))
            features['hist_spread'] = float(np.std(hist))
            
        except Exception as e:
            logger.warning(f"Error in structural features: {e}")
            # Set default values
            features.update({
                'edge_count': 0, 'edge_density': 0, 'contour_count': 0,
                'image_sharpness': 0, 'noise_level': 0, 'blur_score': 0,
                'brightness': 0, 'contrast': 0, 'hist_peak': 0, 'hist_spread': 0
            })
        
        return features
    
    # ============================================================
    # 2. LOGO FEATURES
    # ============================================================
    
    def _extract_logo_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract logo-related features using template matching"""
        features = {}
        
        try:
            if self.logo_template is not None:
                # Template matching
                result = cv2.matchTemplate(gray, self.logo_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                features['logo_match_score'] = float(max_val)
                
                # Get logo region if match is good
                if max_val > 0.5:
                    h, w = self.logo_template.shape
                    logo_region = gray[max_loc[1]:max_loc[1]+h, max_loc[0]:max_loc[0]+w]
                    
                    # Logo area
                    features['logo_area'] = float(h * w)
                    
                    # Logo region features
                    features['logo_mean_intensity'] = float(np.mean(logo_region))
                    features['logo_std_intensity'] = float(np.std(logo_region))
                    
                    # Logo edge density
                    logo_edges = cv2.Canny(logo_region, 50, 150)
                    features['logo_edge_density'] = float(np.sum(logo_edges > 0) / (h * w))
                else:
                    features['logo_area'] = 0
                    features['logo_mean_intensity'] = 0
                    features['logo_std_intensity'] = 0
                    features['logo_edge_density'] = 0
            else:
                # No template provided, try to detect logo region
                features['logo_match_score'] = 0.5  # Neutral score
                features['logo_area'] = 0
                features['logo_mean_intensity'] = 0
                features['logo_std_intensity'] = 0
                features['logo_edge_density'] = 0
                
        except Exception as e:
            logger.warning(f"Error in logo features: {e}")
            features.update({
                'logo_match_score': 0, 'logo_area': 0,
                'logo_mean_intensity': 0, 'logo_std_intensity': 0,
                'logo_edge_density': 0
            })
        
        return features
    
    # ============================================================
    # 3. SEAL FEATURES
    # ============================================================
    
    def _extract_seal_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract seal/circularity features"""
        features = {}
        
        try:
            # Find circular shapes (potential seals)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            circularities = []
            seal_areas = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500:  # Filter small contours
                    continue
                    
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    # Calculate circularity: 4π × Area / Perimeter²
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Circle has circularity close to 1
                    if circularity > 0.5:  # Potential seal
                        circularities.append(circularity)
                        seal_areas.append(area)
            
            if circularities:
                features['seal_circularity'] = float(np.mean(circularities))
                features['seal_circularity_std'] = float(np.std(circularities))
                features['seal_area'] = float(np.mean(seal_areas))
                features['seal_count'] = len(circularities)
                features['max_seal_area'] = float(np.max(seal_areas))
            else:
                features['seal_circularity'] = 0
                features['seal_circularity_std'] = 0
                features['seal_area'] = 0
                features['seal_count'] = 0
                features['max_seal_area'] = 0
            
            # Template matching if seal template provided
            if self.seal_template is not None:
                result = cv2.matchTemplate(gray, self.seal_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                features['seal_match_score'] = float(max_val)
            else:
                features['seal_match_score'] = 0.5
                
        except Exception as e:
            logger.warning(f"Error in seal features: {e}")
            features.update({
                'seal_circularity': 0, 'seal_circularity_std': 0,
                'seal_area': 0, 'seal_count': 0, 'max_seal_area': 0,
                'seal_match_score': 0
            })
        
        return features
    
    # ============================================================
    # 4. SIGNATURE FEATURES
    # ============================================================
    
    def _extract_signature_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract signature features (usually at bottom of certificate)"""
        features = {}
        
        try:
            # Assume signature is in bottom 20% of image
            h, w = gray.shape
            signature_region = gray[int(0.8*h):h, :]
            
            if signature_region.size > 0:
                # Edge density in signature region
                sig_edges = cv2.Canny(signature_region, 30, 100)
                features['signature_edge_density'] = float(np.sum(sig_edges > 0) / signature_region.size)
                
                # Pixel intensity statistics
                features['signature_pixel_mean'] = float(np.mean(signature_region))
                features['signature_pixel_std'] = float(np.std(signature_region))
                
                # Find potential signature contours
                _, sig_thresh = cv2.threshold(signature_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                sig_contours, _ = cv2.findContours(sig_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter contours that might be signatures (based on size and aspect ratio)
                potential_signatures = []
                for cnt in sig_contours:
                    area = cv2.contourArea(cnt)
                    if 500 < area < 10000:  # Typical signature size
                        x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
                        aspect_ratio = w_cnt / h_cnt if h_cnt > 0 else 0
                        if 2 < aspect_ratio < 10:  # Signature is wide
                            potential_signatures.append(cnt)
                
                features['signature_contour_count'] = len(potential_signatures)
                
                if potential_signatures:
                    # Get largest potential signature
                    largest_sig = max(potential_signatures, key=cv2.contourArea)
                    features['signature_area'] = float(cv2.contourArea(largest_sig))
                    features['signature_aspect_ratio'] = float(w_cnt / h_cnt)
                else:
                    features['signature_area'] = 0
                    features['signature_aspect_ratio'] = 0
            else:
                features['signature_edge_density'] = 0
                features['signature_pixel_mean'] = 0
                features['signature_pixel_std'] = 0
                features['signature_contour_count'] = 0
                features['signature_area'] = 0
                features['signature_aspect_ratio'] = 0
            
            # Template matching if signature template provided
            if self.signature_template is not None:
                result = cv2.matchTemplate(signature_region, self.signature_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                features['signature_match_score'] = float(max_val)
            else:
                features['signature_match_score'] = 0.5
                
        except Exception as e:
            logger.warning(f"Error in signature features: {e}")
            features.update({
                'signature_edge_density': 0, 'signature_pixel_mean': 0,
                'signature_pixel_std': 0, 'signature_contour_count': 0,
                'signature_area': 0, 'signature_aspect_ratio': 0,
                'signature_match_score': 0
            })
        
        return features
    
    # ============================================================
    # 5. TEXT & FONT FEATURES (VERY IMPORTANT)
    # ============================================================
    
    def _extract_text_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract text and font-related features using OCR"""
        features = {}
        
        try:
            # Preprocess for better OCR
            # Apply adaptive thresholding for better text extraction
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
            
            # Get OCR data with bounding boxes
            ocr_data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
            
            # Extract full text
            text = ' '.join([ocr_data['text'][i] for i in range(len(ocr_data['text'])) 
                            if ocr_data['text'][i].strip()])
            features['extracted_text_length'] = len(text)
            features['word_count'] = len(text.split())
            
            # Extract confidence scores
            confidences = []
            for conf in ocr_data['conf']:
                try:
                    if conf != '-1':
                        confidences.append(float(conf))
                except:
                    pass
            
            features['ocr_avg_confidence'] = float(np.mean(confidences)) if confidences else 0
            features['ocr_min_confidence'] = float(np.min(confidences)) if confidences else 0
            features['ocr_std_confidence'] = float(np.std(confidences)) if confidences else 0
            
            # Extract character height/width variance from bounding boxes
            heights = []
            widths = []
            line_heights = []
            char_spacings = []
            
            n_boxes = len(ocr_data['text'])
            prev_right = None
            
            for i in range(n_boxes):
                if int(ocr_data['conf'][i]) > 30:  # Filter low confidence
                    word = ocr_data['text'][i].strip()
                    if word:
                        h = ocr_data['height'][i]
                        w = ocr_data['width'][i]
                        x = ocr_data['left'][i]
                        
                        heights.append(h)
                        widths.append(w)
                        
                        # Calculate character spacing
                        if prev_right is not None:
                            spacing = x - prev_right
                            if 0 < spacing < 50:  # Reasonable spacing
                                char_spacings.append(spacing)
                        prev_right = x + w
                        
                        # Group by line (using y-coordinate)
                        y = ocr_data['top'][i]
                        line_heights.append(y)
            
            # Height and width variance
            if heights:
                features['char_height_mean'] = float(np.mean(heights))
                features['char_height_std'] = float(np.std(heights))
                features['char_height_variance'] = float(np.var(heights))
                features['char_height_range'] = float(np.max(heights) - np.min(heights))
            else:
                features['char_height_mean'] = 0
                features['char_height_std'] = 0
                features['char_height_variance'] = 0
                features['char_height_range'] = 0
            
            if widths:
                features['char_width_mean'] = float(np.mean(widths))
                features['char_width_std'] = float(np.std(widths))
                features['char_width_variance'] = float(np.var(widths))
            else:
                features['char_width_mean'] = 0
                features['char_width_std'] = 0
                features['char_width_variance'] = 0
            
            # Character spacing variance
            if char_spacings:
                features['char_spacing_mean'] = float(np.mean(char_spacings))
                features['char_spacing_std'] = float(np.std(char_spacings))
                features['char_spacing_variance'] = float(np.var(char_spacings))
                features['char_spacing_range'] = float(np.max(char_spacings) - np.min(char_spacings))
            else:
                features['char_spacing_mean'] = 0
                features['char_spacing_std'] = 0
                features['char_spacing_variance'] = 0
                features['char_spacing_range'] = 0
            
            # Line spacing variance
            if len(line_heights) > 1:
                line_diffs = np.diff(sorted(line_heights))
                features['line_spacing_mean'] = float(np.mean(line_diffs))
                features['line_spacing_std'] = float(np.std(line_diffs))
                features['line_spacing_variance'] = float(np.var(line_diffs))
            else:
                features['line_spacing_mean'] = 0
                features['line_spacing_std'] = 0
                features['line_spacing_variance'] = 0
            
            # Text region density
            text_region = processed.copy()
            text_region[processed < 128] = 0
            features['text_density'] = float(np.sum(text_region > 0) / text_region.size)
            
            # Stroke density (using morphological operations)
            kernel = np.ones((2, 2), np.uint8)
            dilated = cv2.dilate(processed, kernel, iterations=1)
            eroded = cv2.erode(processed, kernel, iterations=1)
            stroke_region = cv2.absdiff(dilated, eroded)
            features['stroke_density'] = float(np.sum(stroke_region > 0) / stroke_region.size)
            
            # Text region sharpness
            text_sharpness = cv2.Laplacian(processed, cv2.CV_64F).var()
            features['text_region_sharpness'] = float(text_sharpness)
            
            # Character count by type
            features['digit_count'] = sum(c.isdigit() for c in text)
            features['letter_count'] = sum(c.isalpha() for c in text)
            features['uppercase_count'] = sum(c.isupper() for c in text)
            features['space_count'] = text.count(' ')
            
            # Ratios
            features['digit_ratio'] = features['digit_count'] / max(len(text), 1)
            features['uppercase_ratio'] = features['uppercase_count'] / max(features['letter_count'], 1)
            
        except Exception as e:
            logger.warning(f"Error in text features: {e}")
            features.update({
                'extracted_text_length': 0, 'word_count': 0,
                'ocr_avg_confidence': 0, 'ocr_min_confidence': 0, 'ocr_std_confidence': 0,
                'char_height_mean': 0, 'char_height_std': 0, 'char_height_variance': 0, 'char_height_range': 0,
                'char_width_mean': 0, 'char_width_std': 0, 'char_width_variance': 0,
                'char_spacing_mean': 0, 'char_spacing_std': 0, 'char_spacing_variance': 0, 'char_spacing_range': 0,
                'line_spacing_mean': 0, 'line_spacing_std': 0, 'line_spacing_variance': 0,
                'text_density': 0, 'stroke_density': 0, 'text_region_sharpness': 0,
                'digit_count': 0, 'letter_count': 0, 'uppercase_count': 0, 'space_count': 0,
                'digit_ratio': 0, 'uppercase_ratio': 0
            })
        
        return features
    
    # ============================================================
    # 6. FORMAT VALIDATION FEATURES
    # ============================================================
    
    def _extract_format_validation_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract features for validating certificate format"""
        features = {}
        
        try:
            # Get OCR text
            text = pytesseract.image_to_string(gray)
            
            # USN validation (example: 4 letters + 2 digits + 3 digits)
            usn_pattern = r'[A-Z]{4}\d{2}[A-Z]{2}\d{3}|\d{1,3}[-]\d{1,3}[-]\d{1,3}'
            usn_matches = re.findall(usn_pattern, text)
            features['usn_count'] = len(usn_matches)
            features['usn_valid'] = 1.0 if len(usn_matches) >= 1 else 0.0
            
            # Check for suspicious USN patterns (like "1 2 3" spaced out)
            spaced_digits = re.findall(r'\d\s+\d\s+\d', text)
            features['suspicious_usn_pattern'] = float(len(spaced_digits) > 0)
            
            # Date validation
            date_patterns = [
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # DD/MM/YYYY
                r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',   # YYYY/MM/DD
                r'\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{2,4}'
            ]
            
            date_count = 0
            for pattern in date_patterns:
                date_count += len(re.findall(pattern, text, re.IGNORECASE))
            
            features['date_count'] = date_count
            features['date_valid'] = 1.0 if date_count >= 1 else 0.0
            
            # Score validation (marks/grades)
            # Find numbers that could be scores (between 0-100)
            score_pattern = r'\b(\d{1,3})\b'
            potential_scores = re.findall(score_pattern, text)
            
            valid_scores = [int(s) for s in potential_scores if s.isdigit() and 0 <= int(s) <= 100]
            
            if valid_scores:
                features['score_count'] = len(valid_scores)
                features['score_mean'] = float(np.mean(valid_scores))
                features['score_std'] = float(np.std(valid_scores))
                features['score_validity_ratio'] = len(valid_scores) / max(len(potential_scores), 1)
            else:
                features['score_count'] = 0
                features['score_mean'] = 0
                features['score_std'] = 0
                features['score_validity_ratio'] = 0
            
            # Grade validation (A+, A, B+, etc.)
            grade_pattern = r'[A-E][+-]?'
            grades = re.findall(grade_pattern, text)
            features['grade_count'] = len(grades)
            
            # Name format validation (capitalized words)
            name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
            names = re.findall(name_pattern, text)
            features['name_count'] = len(names)
            features['avg_name_length'] = float(np.mean([len(n) for n in names])) if names else 0
            
            # Check for table-like structure (multiple numbers in rows)
            lines = text.split('\n')
            numbers_per_line = [len(re.findall(r'\d+', line)) for line in lines if line.strip()]
            if numbers_per_line:
                features['table_structure_score'] = float(np.std(numbers_per_line) / (np.mean(numbers_per_line) + 1))
            else:
                features['table_structure_score'] = 0
            
            # Check for consistent line lengths (helps detect tampering)
            line_lengths = [len(line) for line in lines if line.strip()]
            if line_lengths:
                features['line_length_mean'] = float(np.mean(line_lengths))
                features['line_length_std'] = float(np.std(line_lengths))
                features['line_length_variance'] = float(np.var(line_lengths))
            else:
                features['line_length_mean'] = 0
                features['line_length_std'] = 0
                features['line_length_variance'] = 0
            
        except Exception as e:
            logger.warning(f"Error in format validation: {e}")
            features.update({
                'usn_count': 0, 'usn_valid': 0, 'suspicious_usn_pattern': 0,
                'date_count': 0, 'date_valid': 0,
                'score_count': 0, 'score_mean': 0, 'score_std': 0, 'score_validity_ratio': 0,
                'grade_count': 0, 'name_count': 0, 'avg_name_length': 0,
                'table_structure_score': 0, 'line_length_mean': 0,
                'line_length_std': 0, 'line_length_variance': 0
            })
        
        return features


# ============================================================
# HELPER FUNCTION FOR BATCH PROCESSING
# ============================================================

def extract_features_batch(image_paths: List[str], 
                          logo_template: Optional[str] = None,
                          seal_template: Optional[str] = None,
                          signature_template: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Extract features from multiple images
    
    Args:
        image_paths: List of paths to certificate images
        logo_template: Path to logo template
        seal_template: Path to seal template
        signature_template: Path to signature template
        
    Returns:
        List of feature dictionaries
    """
    extractor = CertificateFeatureExtractor(
        logo_template_path=logo_template,
        seal_template_path=seal_template,
        signature_template_path=signature_template
    )
    
    all_features = []
    for path in image_paths:
        features = extractor.extract_all_features(path)
        features['image_path'] = path
        all_features.append(features)
        
    return all_features


# ============================================================
# MAIN FUNCTION FOR SINGLE IMAGE PROCESSING
# ============================================================

def extract_features_single(image_path: str,
                           logo_template: Optional[str] = None,
                           seal_template: Optional[str] = None,
                           signature_template: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract features from a single image (main function to use)
    
    Args:
        image_path: Path to certificate image
        logo_template: Path to logo template (optional)
        seal_template: Path to seal template (optional)
        signature_template: Path to signature template (optional)
        
    Returns:
        Dictionary of numeric features
    """
    extractor = CertificateFeatureExtractor(
        logo_template_path=logo_template,
        seal_template_path=seal_template,
        signature_template_path=signature_template
    )
    
    return extractor.extract_all_features(image_path)


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Example 1: Single image processing
    print("="*60)
    print("FEATURE EXTRACTION FOR FAKE CERTIFICATE DETECTION")
    print("="*60)
    
    # Path to your certificate image
    image_path = r"C:\Users\HP\OneDrive\Desktop\certificate_detector\data\genuine\gen1.jpg"  # Change this!
    
    # Optional: Path to template images
    logo_template = r"C:\Users\HP\OneDrive\Desktop\certificate_detector\templates\genuine_logo.png"  # Change if you have
    signature_template = r"C:\Users\HP\OneDrive\Desktop\certificate_detector\templates\genuine_sign.png"  # Change if you have
    
    # Extract features
    print(f"\n📄 Processing: {image_path}")
    features = extract_features_single(
        image_path,
        logo_template=logo_template,
        signature_template=signature_template
    )
    
    # Print results
    print(f"\n✅ Extracted {len(features)} features:")
    print("-"*40)
    
    # Group and print features by category
    categories = {
        'Structural': ['edge_count', 'contour_count', 'image_sharpness', 'noise_level', 'blur_score'],
        'Logo': ['logo_match_score', 'logo_area', 'logo_edge_density'],
        'Seal': ['seal_circularity', 'seal_area', 'seal_count', 'seal_match_score'],
        'Signature': ['signature_edge_density', 'signature_pixel_std', 'signature_match_score'],
        'Text': ['word_count', 'ocr_avg_confidence', 'char_height_variance', 'char_spacing_variance'],
        'Format': ['usn_valid', 'date_valid', 'score_validity_ratio']
    }
    
    for category, feat_names in categories.items():
        print(f"\n{category} Features:")
        for name in feat_names:
            if name in features:
                value = features[name]
                if isinstance(value, float):
                    print(f"  {name:25}: {value:.4f}")
                else:
                    print(f"  {name:25}: {value}")
    
    # Show top suspicious features if any
    suspicious_thresholds = {
        'ocr_avg_confidence': 50,
        'char_spacing_variance': 50,
        'char_height_variance': 30,
        'seal_circularity': 0.3,
        'logo_match_score': 0.5,
        'signature_match_score': 0.4
    }
    
    print("\n🔍 Potential Issues:")
    issues_found = False
    for feat, threshold in suspicious_thresholds.items():
        if feat in features and features[feat] < threshold:
            print(f"  ⚠️  Low {feat}: {features[feat]:.4f} (threshold: {threshold})")
            issues_found = True
    
    if not issues_found:
        print("  ✅ No obvious issues detected")
    
    print("\n" + "="*60)
    
    # Example 2: Save features to CSV (for dataset creation)
    import pandas as pd
    
    # Create a DataFrame with single row
    df = pd.DataFrame([features])
    
    # Save to CSV (remove image_path if you don't want it)
    csv_path = "certificate_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Features saved to: {csv_path}")
    
    # Example 3: Batch processing (uncomment to use)
    """
    image_folder = "path/to/your/images/folder"
    import glob
    image_paths = glob.glob(f"{image_folder}/*.jpg") + glob.glob(f"{image_folder}/*.png")
    
    all_features = extract_features_batch(
        image_paths[:5],  # Process first 5 images
        logo_template=logo_template
    )
    
    # Convert to DataFrame and save
    df_batch = pd.DataFrame(all_features)
    df_batch.to_csv("batch_features.csv", index=False)
    print(f"Batch features saved: {len(df_batch)} images")
    """