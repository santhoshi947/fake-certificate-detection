"""
predict.py — Core prediction module for CertifyX
=================================================
Pipeline:
  1. Feature extraction  2. Unknown/OOD check  3. ML prediction
  4. Template similarity (SSIM)  5. Institution OCR check
  6. SUSPICIOUS override  7. Feature analysis  8. Risk calc  9. Annotation
"""

import os, re
from difflib import SequenceMatcher
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2, numpy as np, pandas as pd, joblib, pytesseract
from skimage.metrics import structural_similarity as ssim
from feature_extractor import extract_features_for_prediction, MODEL_FEATURE_COLUMNS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_BASE = Path(__file__).parent
TEMPLATES_DIR           = _BASE / "templates"
KNOWN_INSTITUTIONS_FILE = _BASE / "known_institutions.txt"

_MODEL_PATHS = [_BASE/"certificate_detection_model.pkl", _BASE/"model.pkl",
                Path("certificate_detection_model.pkl"), Path("model.pkl")]
MODEL = None
for _p in _MODEL_PATHS:
    if _p.exists():
        MODEL = joblib.load(str(_p)); break
if MODEL is None:
    logger.warning("No trained model found.")

TEMPLATE_SIMILARITY_THRESHOLD = 0.50
INSTITUTION_FUZZY_THRESHOLD   = 0.75
SUSPICIOUS_MIN_CRITICAL_FAILS = 2
CONFIDENCE_LOW_THRESHOLD      = 60

CRITICAL_THRESHOLDS: Dict[str, Tuple[float, str, str]] = {
    "ocr_avg_confidence":     (40.0,  "below", "Low OCR confidence — text may be tampered or blurred"),
    "logo_match_score":       (0.35,  "below", "Logo mismatch — logo may be replaced or missing"),
    "seal_circularity":       (0.30,  "below", "Seal shape anomaly — circular seal not detected"),
    "seal_match_score":       (0.30,  "below", "Official seal not recognized"),
    "signature_match_score":  (0.30,  "below", "Signature texture anomaly — may be forged"),
    "signature_edge_density": (0.005, "below", "Signature region appears empty or tampered"),
    "char_height_variance":   (200.0, "above", "Font inconsistency — multiple font sizes used"),
    "char_spacing_variance":  (100.0, "above", "Abnormal character spacing — possible manipulation"),
    "image_sharpness":        (50.0,  "below", "Image quality too low — may be a low-resolution forgery"),
    "date_valid":             (1.0,   "below", "No valid issue date found on certificate"),
    "score_validity_ratio":   (0.3,   "below", "Score values appear suspicious or out of range"),
    "table_structure_score":  (3.5,   "above", "Structural misalignment in score/marks table"),
    "edge_density":           (0.02,  "below", "Very low structural detail — possible blank template"),
}
SOFT_THRESHOLDS: Dict[str, Tuple[float, str, str]] = {
    "word_count":            (5.0,   "below", "Very few words detected — possible blank or foreign template"),
    "extracted_text_length": (20.0,  "below", "Insufficient text content extracted"),
    "line_spacing_variance": (500.0, "above", "Inconsistent line spacing — possible text insertion"),
    "blur_score":            (30.0,  "below", "Image is excessively blurred"),
    "usn_valid":             (1.0,   "below", "USN/ID number missing or in invalid format"),
    "contour_count":         (10.0,  "below", "Too few structural elements — certificate appears sparse"),
}
FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    "ocr_avg_confidence":    "OCR Confidence",
    "logo_match_score":      "Logo Similarity",
    "seal_circularity":      "Seal Shape Score",
    "signature_match_score": "Signature Match",
    "char_height_variance":  "Font Variance",
    "char_spacing_variance": "Char Spacing Variance",
    "image_sharpness":       "Image Sharpness",
    "usn_valid":             "USN/ID Valid",
    "date_valid":            "Date Valid",
    "score_validity_ratio":  "Score Validity",
    "edge_density":          "Edge Density",
    "word_count":            "Word Count",
}

RED=(0,0,220); GREEN=(34,197,94); AMBER=(0,140,255); ORANGE=(0,100,255)


# ── A. TEMPLATE MATCHING ─────────────────────────────────────

def _load_template_images():
    templates = []
    for pat in ("*.jpg","*.jpeg","*.png","*.bmp","*.tiff","*.tif"):
        for p in TEMPLATES_DIR.glob(pat):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None: templates.append((img, p.stem))
    return templates

def check_template_similarity(image):
    templates = _load_template_images()
    if not templates: return 1.0, None, False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    best_score, best_template = 0.0, None
    for tmpl_gray, tmpl_name in templates:
        resized = cv2.resize(tmpl_gray, (w, h))
        g1 = cv2.GaussianBlur(gray,    (5,5), 0)
        g2 = cv2.GaussianBlur(resized, (5,5), 0)
        score, _ = ssim(g1, g2, full=True)
        if score > best_score: best_score, best_template = score, tmpl_name
    return best_score, best_template, True


# ── B. INSTITUTION VERIFICATION ──────────────────────────────

def _load_known_institutions():
    if KNOWN_INSTITUTIONS_FILE.exists():
        with open(KNOWN_INSTITUTIONS_FILE, "r", encoding="utf-8") as f:
            return [l.strip().lower() for l in f if l.strip() and not l.strip().startswith("#")]
    return ["university","college","institute","board","cbse","iit","nit","vtu"]

def extract_institution_from_text(text):
    keywords = ["university","college","institute","board","academy","school",
                "polytechnic","vidyapeetham","vidyalaya","mahavidyalaya"]
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines[:15]:
        if any(kw in line.lower() for kw in keywords): return line
    return None

def check_institution_known(image):
    h = image.shape[0]
    top_crop = image[:int(h*0.35), :]
    gray = cv2.cvtColor(top_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    try:
        text = pytesseract.image_to_string(thresh, config="--psm 6")
    except Exception as e:
        logger.warning(f"OCR institution check failed: {e}"); return True, None
    institution = extract_institution_from_text(text)
    if institution is None: return True, None
    known_list = _load_known_institutions()
    inst_lower = institution.lower()
    if any(kw in inst_lower for kw in known_list): return True, institution
    best_ratio = max((SequenceMatcher(None, kw, inst_lower).ratio() for kw in known_list), default=0.0)
    return best_ratio >= INSTITUTION_FUZZY_THRESHOLD, institution


# ── C. SUSPICIOUS OVERRIDE ────────────────────────────────────

def evaluate_suspicious_override(prediction, image, critical_count):
    suspicious_reasons = []
    template_score, best_template, check_done = check_template_similarity(image)
    template_failed = check_done and template_score < TEMPLATE_SIMILARITY_THRESHOLD
    if template_failed:
        suspicious_reasons.append(
            f"Layout similarity {template_score:.0%} — below accepted threshold ({int(TEMPLATE_SIMILARITY_THRESHOLD*100)}%)")
    institution_known, institution_name = check_institution_known(image)
    if institution_known is False:
        suspicious_reasons.append(f"Unrecognized issuer: '{institution_name}'")
    any_check_failed = template_failed or (institution_known is False)
    is_suspicious = (prediction == "GENUINE" and any_check_failed
                     and critical_count >= SUSPICIOUS_MIN_CRITICAL_FAILS)
    return is_suspicious, template_score, best_template, institution_known, institution_name, suspicious_reasons


# ── D. FEATURE ANALYSIS ───────────────────────────────────────

def _check_threshold(value, threshold, direction):
    try: v = float(value)
    except: return False
    return (direction=="below" and v<threshold) or (direction=="above" and v>threshold)

def analyse_features(feature_dict, prediction):
    critical_fails, soft_fails = [], []
    for fname, (thr, direction, msg) in CRITICAL_THRESHOLDS.items():
        val = feature_dict.get(fname)
        if val is not None and _check_threshold(val, thr, direction):
            critical_fails.append(msg)
    for fname, (thr, direction, msg) in SOFT_THRESHOLDS.items():
        val = feature_dict.get(fname)
        if val is not None and _check_threshold(val, thr, direction):
            soft_fails.append(msg)
    critical_count = len(critical_fails)
    force_show = critical_count >= 3
    if prediction in ("FAKE","SUSPICIOUS") or force_show:
        return critical_fails + soft_fails, critical_count, False
    return [], critical_count, True


# ── E. HELPERS ────────────────────────────────────────────────

def assess_unknown_certificate(feature_dict):
    ocr_conf  = float(feature_dict.get("ocr_avg_confidence", 50))
    word_cnt  = float(feature_dict.get("word_count", 20))
    edge_den  = float(feature_dict.get("edge_density", 0.05))
    sharpness = float(feature_dict.get("image_sharpness", 80))
    if ocr_conf < 8 and word_cnt < 3:
        return True, "Extremely low text content — image does not resemble a certificate"
    if edge_den < 0.003:
        return True, "Almost no structural content — image may be blank or corrupted"
    if sharpness < 5:
        return True, "Image is too blurry — please upload a clearer photo"
    if word_cnt < 2:
        return True, "No readable text found — not a document image"
    return False, ""

def calculate_risk(confidence_fake, issue_count, is_unknown, prediction, critical_count):
    forgery_pct = int(min(100, confidence_fake * 100 + issue_count * 3))
    if is_unknown: return "HIGH", max(forgery_pct, 95)
    if prediction == "SUSPICIOUS": return "HIGH", max(forgery_pct, 75)
    if prediction == "GENUINE":
        if critical_count >= 5: return "MEDIUM", max(forgery_pct, 45)
        return "LOW", min(forgery_pct, 35)
    if confidence_fake >= 0.80 or issue_count >= 5: return "HIGH", forgery_pct
    if confidence_fake >= 0.55 or issue_count >= 2: return "MEDIUM", forgery_pct
    return "LOW", forgery_pct

def get_key_feature_scores(feature_dict):
    result = {}
    for key, display_name in FEATURE_DISPLAY_NAMES.items():
        val = feature_dict.get(key)
        if val is not None:
            try: result[display_name] = round(float(val), 4)
            except: pass
    return result


# ── F. ACCURATE REGION DETECTION & ANNOTATION ─────────────────

def _clamp(val, lo, hi):
    return max(lo, min(hi, val))

def _draw_region(img, x1, y1, x2, y2, color, label, font, fscale, thick):
    """Draw a clean, tight bounding box with a readable pill label above the region."""
    ih, iw = img.shape[:2]
    x1, y1, x2, y2 = _clamp(x1,0,iw-1), _clamp(y1,0,ih-1), _clamp(x2,0,iw-1), _clamp(y2,0,ih-1)
    if x2 <= x1 or y2 <= y1:
        return
    # subtle tinted fill
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(ov, 0.12, img, 0.88, 0, img)
    # solid border
    cv2.rectangle(img, (x1, y1), (x2, y2), color, max(2, thick))
    # pill label — prefer to place above, fall back to inside top
    (tw, th), _ = cv2.getTextSize(label, font, fscale, thick)
    pad = 4
    lx = _clamp(x1, 0, iw - tw - 2*pad - 2)
    ly = y1 - pad - 2          # try above the box
    if ly - th < 0:             # would clip top → put inside top
        ly = y1 + th + pad
    cv2.rectangle(img, (lx, ly - th - pad), (lx + tw + 2*pad, ly + pad), color, -1)
    cv2.putText(img, label, (lx + pad, ly), font, fscale, (255,255,255), thick, cv2.LINE_AA)


def _detect_seal_circles(gray, h, w):
    """HoughCircles — finds actual circular seals."""
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=min(h, w) // 4,
        param1=60, param2=35,
        minRadius=int(min(h, w) * 0.04),
        maxRadius=int(min(h, w) * 0.22),
    )
    if circles is None:
        return []
    return [(int(cx), int(cy), int(r)) for cx, cy, r in np.round(circles[0]).astype(int)]


def _detect_logo_region(gray, h, w):
    """Find the highest-edge-density cell in the top 28% — likely where the logo is."""
    top = gray[:int(h * 0.28), :]
    th, tw = top.shape
    cell_w = max(tw // 8, 1)
    edges = cv2.Canny(top, 30, 100)
    best_x, best_d = 0, 0.0
    for i in range(tw // cell_w):
        cell = edges[:, i * cell_w: (i + 1) * cell_w]
        d = np.sum(cell > 0) / max(cell.size, 1)
        if d > best_d:
            best_d, best_x = d, i * cell_w
    if best_d < 0.008:
        return None
    return (best_x, 0, min(best_x + cell_w * 2, tw), th)


def _detect_signature_region(gray, h, w):
    """Find the most signature-like contour in the bottom 28%."""
    bottom = gray[int(h * 0.72):, :]
    bh, bw = bottom.shape
    _, thresh = cv2.threshold(bottom, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 150:
            continue
        bx, by, cw, ch_c = cv2.boundingRect(cnt)
        aspect = cw / max(ch_c, 1)
        if 1.8 < aspect < 18 and 150 < area < bw * bh * 0.25:
            score = area * min(aspect, 10)
            if score > best_score:
                best_score = score
                best = (bx, int(h * 0.72) + by, bx + cw, int(h * 0.72) + by + ch_c)
    return best


def _get_word_anomalies(gray, warn_color):
    """
    Return list of (x1,y1,x2,y2,reason) for words that look tampered:
      - Very low OCR confidence (< 35)
      - Unusual font height vs mean
    Limit to 12 boxes maximum so image stays readable.
    """
    try:
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT,
                                         config="--psm 6")
    except Exception:
        return []

    n = len(data["text"])
    valid_heights = []
    for i in range(n):
        txt = str(data["text"][i]).strip()
        try:
            conf = float(data["conf"][i])
        except Exception:
            continue
        if txt and conf > 15 and data["height"][i] > 4:
            valid_heights.append(data["height"][i])

    mean_h = float(np.mean(valid_heights)) if valid_heights else 0.0
    anomalies = []
    for i in range(n):
        txt = str(data["text"][i]).strip()
        if not txt:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            continue
        wd, ht = data["width"][i], data["height"][i]
        if wd < 4 or ht < 4:
            continue
        wx1, wy1 = data["left"][i], data["top"][i]
        reason = None
        if 0 <= conf < 35:
            reason = f"low-conf {int(conf)}%"
        elif mean_h > 0 and (ht > mean_h * 2.0 or ht < mean_h * 0.35):
            reason = "font-size anomaly"
        if reason:
            anomalies.append((wx1, wy1, wx1 + wd, wy1 + ht, reason))

    # Dedupe/limit — keep most extreme examples
    return anomalies[:12]


def draw_suspicious_regions(image, feature_dict, prediction, confidence, critical_count):
    """
    Clean forensic annotation:
      1. Actual seal detected via HoughCircles → circle drawn on it
      2. Logo region via edge-density analysis → box on detected area
      3. Signature via contour analysis → box on detected strokes
      4. Date/USN/Score region → bottom-right box with specific failing checks named
      5. Word-level anomalies (low OCR conf + font-size) → underline + small reason tag
      6. Dark verdict strip at bottom
    For GENUINE (no mass failures) → all green, no word-level anomalies drawn.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]
    gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fscale = max(0.38, min(0.65, w / 1300))
    thick  = max(1, int(w / 700))

    force_green = (prediction == "GENUINE" and critical_count < 3)
    warn_color  = ORANGE if prediction == "SUSPICIOUS" else RED

    def _col(ok): return GREEN if (force_green or ok) else warn_color
    def _lbl(ok, t, f): return t if (force_green or ok) else f

    # ──── 1. SEAL ─────────────────────────────────────────────────
    seal_ok = (float(feature_dict.get("seal_circularity", 0)) >= 0.30
               and float(feature_dict.get("seal_match_score", 0.5)) >= 0.30)
    seal_circles = _detect_seal_circles(gray, h, w)

    if seal_circles:
        for cx, cy, r in seal_circles[:2]:          # draw max 2 seals
            color = GREEN if (force_green or seal_ok) else warn_color
            cv2.circle(annotated, (cx, cy), r + 3, color, thick + 1)
            lbl = "Seal: OK" if (force_green or seal_ok) else "Seal: Irregular Shape"
            _draw_region(annotated, cx-r, cy-r, cx+r, cy+r, color, lbl, font, fscale, thick)
    else:
        color = GREEN if force_green else warn_color
        lbl   = "Seal: Present" if force_green else "Seal: NOT DETECTED"
        _draw_region(annotated, int(w*0.70), int(h*0.01), int(w*0.99), int(h*0.24),
                     color, lbl, font, fscale, thick)

    # ──── 2. LOGO / HEADER ────────────────────────────────────────
    logo_score = float(feature_dict.get("logo_match_score", 0.5))
    logo_ok    = logo_score >= 0.35
    logo_bbox  = _detect_logo_region(gray, h, w)
    if logo_bbox:
        x1, y1, x2, y2 = logo_bbox
        color = _col(logo_ok)
        lbl   = _lbl(logo_ok, "Logo: Verified", f"Logo: Mismatch ({logo_score:.0%})")
        _draw_region(annotated, x1, y1, x2, y2, color, lbl, font, fscale, thick)
    else:
        color = GREEN if force_green else warn_color
        lbl   = "Header: OK" if force_green else "Logo/Header: Missing"
        _draw_region(annotated, 0, 0, int(w*0.40), int(h*0.22), color, lbl, font, fscale, thick)

    # ──── 3. SIGNATURE ────────────────────────────────────────────
    sig_ok   = (float(feature_dict.get("signature_match_score", 0.5)) >= 0.30
                and float(feature_dict.get("signature_edge_density", 0)) >= 0.005)
    sig_bbox = _detect_signature_region(gray, h, w)
    if sig_bbox:
        x1, y1, x2, y2 = sig_bbox
        color = _col(sig_ok)
        lbl   = _lbl(sig_ok, "Signature: Found", "Signature: Anomalous Strokes")
        _draw_region(annotated, x1, y1, x2, y2, color, lbl, font, fscale, thick)
    else:
        color = GREEN if force_green else warn_color
        lbl   = "Signature: OK" if force_green else "Signature: NOT FOUND"
        _draw_region(annotated, int(w*0.03), int(h*0.76), int(w*0.52), h-22,
                     color, lbl, font, fscale, thick)

    # ──── 4. DATE / SCORE / USN FIELDS (bottom-right) ─────────────
    date_valid  = float(feature_dict.get("date_valid", 1))
    score_ratio = float(feature_dict.get("score_validity_ratio", 1))
    usn_valid   = float(feature_dict.get("usn_valid", 1))     # advisory only
    table_score = float(feature_dict.get("table_structure_score", 0))
    # USN is advisory — only partial weight, not primary check
    table_ok    = date_valid >= 1.0 and score_ratio >= 0.3 and table_score <= 3.5
    color = _col(table_ok)
    sub_issues = []
    if not force_green:
        if date_valid < 1.0:   sub_issues.append("No Date Found")
        if score_ratio < 0.3:  sub_issues.append("Invalid Scores")
        if table_score > 3.5:  sub_issues.append("Table Misaligned")
        if usn_valid < 1.0:    sub_issues.append("USN Format Odd")   # soft advisory
    lbl = ("Scores/Date: OK" if (force_green or table_ok)
           else ("Data: " + " | ".join(sub_issues) if sub_issues else "Data: CHECK"))
    _draw_region(annotated, int(w*0.55), int(h*0.74), w-2, h-22,
                 color, lbl, font, fscale, thick)

    # ──── 5. WORD-LEVEL ANOMALIES (only for FAKE / SUSPICIOUS) ─────
    # Drawn as colored underlines so they don't obscure the text
    if not force_green:
        ocr_conf = float(feature_dict.get("ocr_avg_confidence", 80))
        font_var = float(feature_dict.get("char_height_variance", 0))
        if ocr_conf < 40 or font_var > 200:
            anomalies = _get_word_anomalies(gray, warn_color)
            uc = (0, 180, 255)   # bright yellow-orange underline
            for (ax1, ay1, ax2, ay2, reason) in anomalies:
                # Draw underline only (much cleaner than a full box)
                uy = min(ay2 + 2, h - 2)
                cv2.line(annotated, (ax1, uy), (ax2, uy), uc, max(2, thick))
                # small reason tag at top-left of word
                cv2.putText(annotated, reason, (ax1, max(ay1 - 3, 10)),
                            font, fscale * 0.50, uc, 1, cv2.LINE_AA)

    # ──── 6. VERDICT STRIP ───────────────────────────────────────
    strip_col  = GREEN if prediction == "GENUINE" else (ORANGE if prediction == "SUSPICIOUS" else RED)
    icon_map   = {"GENUINE": "[PASS]", "FAKE": "[FAIL]", "SUSPICIOUS": "[WARN]"}
    wm_txt     = f"CertifyX {icon_map.get(prediction,'?')} {prediction} — {confidence}% confidence"
    (ww, wh), _ = cv2.getTextSize(wm_txt, font, fscale * 0.82, thick)
    sh = wh + 16
    cv2.rectangle(annotated, (0, h - sh), (w, h), (10, 10, 14), -1)
    cv2.putText(annotated, wm_txt, (8, h - 8), font, fscale * 0.82, strip_col, thick, cv2.LINE_AA)



# ── G. MAIN PIPELINE ─────────────────────────────────────────



def predict_certificate(image_path: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "prediction": None, "confidence": 0, "confidence_fake": 0.0,
        "detected_issues": [], "critical_fail_count": 0, "suppress_issues": False,
        "is_low_confidence": False, "risk_level": "LOW", "forgery_probability": 0,
        "is_unknown": False, "unknown_reason": "",
        "template_score": 1.0, "best_template": None, "templates_checked": False,
        "institution_name": None, "institution_known": None, "suspicious_reasons": [],
        "annotated_image": None, "feature_dict": {}, "key_scores": {}, "error": None,
    }
    try:
        if MODEL is None: raise RuntimeError("Model not loaded.")
        feature_vector, feature_dict = extract_features_for_prediction(image_path)
        result["feature_dict"] = feature_dict
        result["key_scores"]   = get_key_feature_scores(feature_dict)

        img = cv2.imread(image_path)
        if img is None: raise RuntimeError(f"Could not read image: {image_path}")

        is_unknown, unknown_reason = assess_unknown_certificate(feature_dict)
        result["is_unknown"] = is_unknown; result["unknown_reason"] = unknown_reason

        if is_unknown:
            issues, crit_count, _ = analyse_features(feature_dict, "FAKE")
            issues.insert(0, f"Unknown format: {unknown_reason}")
            result.update({
                "prediction": "FAKE", "confidence": 95, "confidence_fake": 0.95,
                "detected_issues": issues, "critical_fail_count": crit_count,
                "suppress_issues": False, "is_low_confidence": False,
                "risk_level": "HIGH", "forgery_probability": 95,
                "annotated_image": draw_suspicious_regions(img, feature_dict, "FAKE", 95, crit_count),
            })
            return result

        feature_df = pd.DataFrame(feature_vector, columns=MODEL_FEATURE_COLUMNS)
        proba = MODEL.predict_proba(feature_df)[0]
        confidence_fake    = float(proba[0])
        confidence_genuine = float(proba[1])
        predicted_label    = MODEL.predict(feature_df)[0]

        if predicted_label == 1:
            ml_prediction = "GENUINE"; confidence = int(round(confidence_genuine * 100))
        else:
            ml_prediction = "FAKE";    confidence = int(round(confidence_fake * 100))

        result["confidence_fake"]    = confidence_fake
        result["is_low_confidence"]  = confidence < CONFIDENCE_LOW_THRESHOLD

        issues, critical_count, suppress = analyse_features(feature_dict, ml_prediction)

        (is_suspicious, template_score, best_template,
         institution_known, institution_name, suspicious_reasons
        ) = evaluate_suspicious_override(ml_prediction, img, critical_count)

        result["template_score"]     = template_score
        result["best_template"]      = best_template
        result["templates_checked"]  = len(_load_template_images()) > 0
        result["institution_name"]   = institution_name
        result["institution_known"]  = institution_known
        result["suspicious_reasons"] = suspicious_reasons

        if is_suspicious:
            prediction = "SUSPICIOUS"
            issues, critical_count, suppress = analyse_features(feature_dict, "SUSPICIOUS")
        else:
            prediction = ml_prediction

        result["prediction"]          = prediction
        result["confidence"]          = confidence
        result["detected_issues"]     = issues
        result["critical_fail_count"] = critical_count
        result["suppress_issues"]     = suppress

        risk_level, forgery_pct = calculate_risk(
            confidence_fake, len(issues), is_unknown, prediction, critical_count)
        result["risk_level"]          = risk_level
        result["forgery_probability"] = forgery_pct
        result["annotated_image"]     = draw_suspicious_regions(
            img, feature_dict, prediction, confidence, critical_count)

    except Exception as exc:
        logger.error(f"Prediction failed: {exc}", exc_info=True)
        result["error"] = str(exc)
    return result