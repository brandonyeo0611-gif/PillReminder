import cv2
import numpy as np
from PIL import Image

def preprocess_image(image_path):
    # Read image
    img = cv2.imread(image_path)
    
    # Upscale for better OCR (2x is sweet spot)
    scale = 2
    h, w = img.shape[:2]
    img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for contrast enhancement (helps with faded images)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Denoise with balanced parameters
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Adaptive thresholding with tuned block size
    thresh = cv2.adaptiveThreshold(denoised, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 5)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Deskew (straighten) image
    try:
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(cv2.convexHull(coords))[2]
            if angle < -45:
                angle = angle + 90
            h, w = thresh.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(thresh, M, (w, h))
            return rotated
    except:
        pass
    
    return thresh

import pytesseract
from PIL import Image

# Option 1: Tesseract (Open-source, free)
def extract_text_tesseract(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text


import re
from typing import Dict

def parse_prescription_label(ocr_text: str) -> Dict:
    """
    Parse OCR text and extract medication information.
    Handles multiple formats:
    - Format 1: "... TAB (MEDICATION_NAME)"
    - Format 2: "MedicationName 50mg Tablets"
    """
    
    # Initialize result dictionary
    result = {
        "medication_name": None,
        "dosage": None,
        "instructions": None,
        "patient_name": None,
        "doctor_name": None,
        "pharmacy": None,
        "refills": None,
        "quantity": None
    }
    
    # Medication name — try multiple patterns, prioritize specific formats
    med_patterns = [
        # Format: "MedicationName Digits mg/MG Cap/Tablet" e.g. "Tacrolimus 1mg Cap"
        r"([A-Z][a-z]+)\s+\d+\s*(?:mg|MG)\s+(?:Cap|Tablet|Tab|Capsule|TABLETS|tablets?|TABS?)",
        # Format: Full med name before "MG TAB" - "NIFEDIPINE LA 60MG TAB"
        r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)\s+\d+\s*(?:MG|mg)\s+(?:TAB|CAP|TABLET|CAPSULE)",
        # Format: "WORD+CAPSULE/TABLET DIGITS MG" e.g. "AMOXICICAPSULE 500 MG" - use greedy match
        r"([A-Z][A-Z]+?[aeiou][A-Z]*)\s*(?:CAPSULE|TABLET|CAP|TAB)\s+\d+\s*(?:MG|mg)",
        # Format: Text in parentheses (AUGMENTIN) - lowest priority
        r"\(([A-Z][A-Z0-9\-\s]+?)\)",
    ]
    for pattern in med_patterns:
        med_match = re.search(pattern, ocr_text, re.IGNORECASE)
        if med_match:
            med = med_match.group(1).strip() if med_match.lastindex == 1 else med_match.group(1).strip()
            # Filter out single-letter matches and common false positives
            # Also exclude multi-line results (containing \n)
            if len(med) > 2 and '\n' not in med and not any(kw in med.upper() for kw in ['TAKE', 'PHARMACY', 'DOCTOR', 'PATIENT', 'SENTENCE', 'ACCORD', 'HEALTHCARE', 'KEEP', 'AWAY', 'FROM']):
                result["medication_name"] = med
                break
    
    # Dosage extraction (e.g., 500mg, 2ml, 10 TABLET, 50mg Tablets)
    dosage_pattern = r"(\d+)\s*(?:mg|ml|mcg|g|tablet|tab|capsule|cap|IU|units?)"
    dosage_match = re.search(dosage_pattern, ocr_text, re.IGNORECASE)
    if dosage_match:
        full_match = re.search(r"(\d+(?:\.\d+)?)\s*(mg|ml|mcg|g|tablet|tab|capsule|cap|IU|units?)", ocr_text, re.IGNORECASE)
        if full_match:
            result["dosage"] = f"{full_match.group(1)} {full_match.group(2)}".upper()
    
    # Instructions — try multiple patterns, handle multi-line and OCR errors
    instr_patterns = [
        # Pattern 1: "Take 500mg three times (3x) a day for five days" - match across lines to period or "days"
        r"(?:Take|TAKE|TAME|TAXE|TATE)\s+(\d+\s*mg[\s\w\(\)\-\.,']*?days?\.?)",
        # Pattern 2: "Take ... morning ... bedtime/evening" (allow heavy OCR noise)
        r"(?:Take|TAKE|TAME|TAXE|TATE)\s+(.{0,160}?(?:moming|morning).{0,160}?(?:bedtime|evening|night))",
        # Pattern 3: "Take 1 tablet 2 times a day" with OCR errors like THES/ADAY
        r"(?:Take|TAKE|TAME|TAXE|TATE)\s+([A-Za-z0-9\s\-\.',']+?(?:times|THES|daily|aday|a\s+day|day))",
        # Pattern 4: "Take 500mg three times ... days"
        r"(?:Take|TAKE|TAME|TAXE|TATE)\s+([0-9]+\s*mg[\w\s,\.\-—()]*?(?:times|THES|daily|aday|a\s+day)[\w\s,\.\-—()]*?(?:days?)?)",
        # Pattern 5: Fallback: line with dosage + schedule words
        r"(\d+\s+(?:tablet|pill|capsule|tab|cap)[s]?\s+[\w\s,\.\-—]*?(?:morning|moming|evening|night|bedtime|day|times|daily)[\w\s,\.\-—]*)",
    ]
    for pattern in instr_patterns:
        instr_match = re.search(pattern, ocr_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if instr_match:
            instr = instr_match.group(1).strip()
            # Clean up extra whitespace, newlines, dashes, and OCR garbage
            instr = re.sub(r'[\s—\-|]+', ' ', instr).strip()
            instr = instr.replace('moming', 'morning').replace('aday', 'a day').replace('THES', 'TIMES')
            # Remove trailing numbers/garbage that are OCR errors (e.g., ", 4" at the end)
            instr = re.sub(r',\s*\d+\s*$', '', instr).strip()
            result["instructions"] = instr
            break
    
    # Patient name — look for name-like pattern at start, or explicit "Name:" label
    patient_patterns = [
        # Format: "Patient Name: Olivia Wilson" (explicit label with colon)
        r"Patient\s+Name:\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
        # Format: "TAN AH SENG" (all caps, 2-3 word name before identifier like SXXXX/DOB)
        r"^([A-Z]{3,}(?:\s+[A-Z]{2,}){1,2})\s*$",
        # Format: "JANE SMITH PHARMACY" (all caps at start, then pharmacy)
        r"^([A-Z]{2,}(?:\s+[A-Z]{2,})+)\s+(?:PHARMACY|pharmacy|DOCTOR|doctor)",
        # Format: "Name * 2" (with asterisk separator)
        r"^([A-Z][a-z]+\s+[A-Z][a-z]+)\s+\*",
        # Explicit label: "Patient|Name|For"
        r"(?:Patient|Name|For)[\s:]*([A-Z][A-Za-z\s\-]+?)(?=\n|Age|DOB)",
    ]
    for pattern in patient_patterns:
        patient_match = re.search(pattern, ocr_text, re.IGNORECASE | re.MULTILINE)
        if patient_match:
            name = patient_match.group(1).strip()
            name = re.sub(r"^Sentence\s+", "", name, flags=re.IGNORECASE)
            if len(name) > 3 and not any(kw in name.upper() for kw in ['TABLETS', 'TAKE', 'AUGMENTIN', 'ACCORD', 'TACROLIMUS', 'PHARMACY', 'CAPSULE', 'MEDICINE', 'HOSPITAL', 'BLOOD', 'PRESSURE', 'FOR']):
                result["patient_name"] = name
                break
    
    # Doctor name
    doctor_patterns = [
        r"(?:Consult\s+doctor|Dr\.?|Doctor)[\s:]*([A-Za-z\s\.\-]+?)(?=\n|$)",
    ]
    for pattern in doctor_patterns:
        doctor_match = re.search(pattern, ocr_text, re.IGNORECASE)
        if doctor_match:
            name = doctor_match.group(1).strip()
            if len(name) > 2:
                result["doctor_name"] = name
                break
    
    # Quantity (number of tablets/pills) — look for patterns like "10 TABLETS" or "Take 2 pills"
    quantity_patterns = [
        r"(\d+)\s+(?:TABLETS|TAB|TABLET|capsules?|CAPSULES?|pills?|PILLS?)",
        r"Take\s+(\d+)\s+(?:pills?|tablets?|capsules?)",
    ]
    for pattern in quantity_patterns:
        qty_match = re.search(pattern, ocr_text, re.IGNORECASE)
        if qty_match:
            result["quantity"] = qty_match.group(1)
            break
    
    # Fallback: explicit Qty label
    if not result["quantity"]:
        qty_pattern = r"(?:Qty|Quantity|Num|#)[\s:]*(\d+)"
        qty_match = re.search(qty_pattern, ocr_text, re.IGNORECASE)
        if qty_match:
            result["quantity"] = qty_match.group(1)
    
    # Refills
    refills_pattern = r"(?:Refills?)[\s:]*(\d+)"
    refills_match = re.search(refills_pattern, ocr_text, re.IGNORECASE)
    if refills_match:
        result["refills"] = refills_match.group(1)
    
    return result

import argparse
from typing import Optional

try:
    from drug_validation import validate_medication  # optional external validator
except Exception:
    validate_medication = None


def analyze_prescription_label(image_path: str, save_preprocessed: Optional[str] = None, debug: bool = False):
    """
    Complete prescription label analysis pipeline using the local helpers
    defined in this file. If `validate_medication` is available it will be used;
    otherwise validation is skipped.
    """
    print("1. Preprocessing image...")
    processed_img = preprocess_image(image_path)

    if save_preprocessed:
        # processed_img is a numpy array (grayscale) — convert to PIL and save
        Image.fromarray(processed_img).save(save_preprocessed)

    print("2. Extracting text via OCR...")
    # Use the in-file OCR helper (works with PIL Image)
    pil_img = Image.fromarray(processed_img)
    
    # Try with better OCR config (PSM 6 = single column, OEM 3 = best accuracy)
    config = '--psm 6 --oem 3'
    ocr_text = pytesseract.image_to_string(pil_img, config=config)
    
    if debug:
        print(f"Raw OCR output:\n{ocr_text}\n")

    print("3. Parsing medication information...")
    medication_data = parse_prescription_label(ocr_text)

    # Fallback OCR pass if any critical field is missing
    if not medication_data.get("instructions") or not medication_data.get("medication_name"):
        try:
            alt_img = cv2.imread(image_path)
            if alt_img is not None:
                # Strategy 1: Alt preprocessing with PSM 11
                alt_gray = cv2.cvtColor(alt_img, cv2.COLOR_BGR2GRAY)
                alt_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(alt_gray)
                alt_pil = Image.fromarray(alt_clahe)
                alt_config = '--psm 11 --oem 3'
                alt_text = pytesseract.image_to_string(alt_pil, config=alt_config)
                
                # Strategy 2: Raw image without preprocessing
                raw_pil = Image.open(image_path)
                raw_text = pytesseract.image_to_string(raw_pil, config='--psm 6 --oem 3')
                
                # Strategy 3: 4x upscaling (most effective for small/poor quality images)
                h, w = alt_img.shape[:2]
                upscaled4x = cv2.resize(alt_img, (w * 4, h * 4), interpolation=cv2.INTER_LANCZOS4)
                up4x_gray = cv2.cvtColor(upscaled4x, cv2.COLOR_BGR2GRAY)
                up4x_pil = Image.fromarray(up4x_gray)
                up4x_text = pytesseract.image_to_string(up4x_pil, config='--psm 6 --oem 3')
                
                # Strategy 4: 4x upscaling with CLAHE
                up4x_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(up4x_gray)
                up4x_clahe_pil = Image.fromarray(up4x_clahe)
                up4x_clahe_text = pytesseract.image_to_string(up4x_clahe_pil, config='--psm 6 --oem 3')
                
                if debug:
                    print(f"Alt OCR output:\n{alt_text}\n")
                    print(f"Raw OCR (no preprocess) output:\n{raw_text}\n")
                    print(f"4x upscaled output:\n{up4x_text}\n")
                    print(f"4x upscaled + CLAHE output:\n{up4x_clahe_text}\n")
                
                combined_text = f"{ocr_text}\n{alt_text}\n{raw_text}\n{up4x_text}\n{up4x_clahe_text}"
                alt_data = parse_prescription_label(combined_text)
                for key in ["instructions", "medication_name", "patient_name"]:
                    if not medication_data.get(key) and alt_data.get(key):
                        medication_data[key] = alt_data.get(key)
        except Exception as e:
            if debug:
                print(f"Fallback OCR error: {e}")
            pass

    if medication_data.get("medication_name") and validate_medication:
        try:
            medication_data["validation"] = validate_medication(medication_data["medication_name"])
        except Exception:
            medication_data["validation"] = None

    print("\n✅ Results:")
    print(medication_data)

    return medication_data


def _build_cli():
    p = argparse.ArgumentParser(description="Prescription label OCR + parsing")
    p.add_argument("--image", "-i", required=True, help="Path to the image file to analyze")
    p.add_argument("--save-preprocessed", "-s", help="Optional path to save the preprocessed image")
    p.add_argument("--json-out", "-j", help="Optional path to save output as JSON")
    p.add_argument("--debug", "-d", action="store_true", help="Print raw OCR text for debugging")
    return p


if __name__ == "__main__":
    import json
    import os
    parser = _build_cli()
    args = parser.parse_args()
    result = analyze_prescription_label(args.image, args.save_preprocessed, debug=args.debug)
    
    # Extract only: patient_name, medication_name, instructions
    output = {
        "patient_name": result.get("patient_name"),
        "medication_name": result.get("medication_name"),
        "instructions": result.get("instructions")
    }
    
    # Print to stdout
    print("\n--- SUMMARY ---")
    print(json.dumps(output, indent=2))
    
    # Determine output file path
    if args.json_out:
        output_path = args.json_out
    else:
        # Default: save to label_recognition/results/ with same filename but .json extension
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        results_dir = os.path.join(os.path.dirname(args.image), "results")
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, f"{base_name}.json")
    
    # Save to JSON file
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Saved to {output_path}")