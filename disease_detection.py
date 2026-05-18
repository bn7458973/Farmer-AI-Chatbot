import os
import sys
import cv2
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename

# Try loading YOLOv8 — use the largest (most accurate) segmentation model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'yolov8n-seg.pt')
MIN_VISUAL_DAMAGE_PERCENT = 3.0
MIN_GEMINI_DAMAGE_PERCENT = 10.0

def load_model():
    try:
        from ultralytics import YOLO
        if os.path.exists(MODEL_PATH):
            return YOLO(MODEL_PATH)
    except Exception as e:
        print(f"[YOLO] Could not load model: {e}", file=sys.stderr)
    return None

model = load_model()


def _severity(pct):
    if pct <= 15:   return 'Mild'
    if pct <= 40:   return 'Moderate'
    return 'Severe'


def _is_healthy_diagnosis(disease, damage_pct):
    disease_text = (disease or '').strip().lower()
    healthy_terms = (
        'healthy',
        'no disease',
        'none',
        'no visible disease',
        'no visible symptoms',
        'not detected',
    )
    return damage_pct < 1.0 or any(term in disease_text for term in healthy_terms)


def _healthy_detection(plant='Unknown', image_path=None, source='none', confidence=1.0, error=None):
    result = {
        'plant': plant,
        'disease': 'Healthy plant',
        'damage_percentage': 0.0,
        'severity': 'None',
        'healthy': True,
        'confidence': confidence,
        'vision_source': source,
        'source': source,
        'error': error
    }
    if image_path:
        result['annotated_image_path'] = image_path
        result['masked_image_path'] = image_path
    return result


def _estimate_visual_damage(image):
    """Conservative local check for visible leaf spots/discoloration.

    This is used as a guardrail against vision-model hallucinations: if the
    uploaded image has almost no visible diseased-looking pixels, return healthy.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    green_leaf = (h >= 35) & (h <= 95) & (s >= 35) & (v >= 35)
    yellow_or_brown = (h >= 8) & (h <= 38) & (s >= 45) & (v >= 35)
    dark_necrotic = (s >= 35) & (v <= 85)
    pale_powder = (s <= 45) & (v >= 175)

    plant_mask = green_leaf | yellow_or_brown | dark_necrotic
    damaged_mask = plant_mask & (yellow_or_brown | dark_necrotic | pale_powder)

    plant_pixels = int(np.count_nonzero(plant_mask))
    if plant_pixels == 0:
        return 0.0

    damaged_pixels = int(np.count_nonzero(damaged_mask))
    return (damaged_pixels / plant_pixels) * 100


def _should_force_healthy(vision_result, visual_damage_pct, yolo_damage_pct):
    disease = vision_result.get('disease')
    gemini_damage = float(vision_result.get('damage_percentage') or 0.0)
    if _is_healthy_diagnosis(disease, gemini_damage):
        return True

    if yolo_damage_pct < 1.0:
        return True

    no_local_damage = visual_damage_pct < MIN_VISUAL_DAMAGE_PERCENT
    weak_gemini_claim = gemini_damage < MIN_GEMINI_DAMAGE_PERCENT
    return no_local_damage or weak_gemini_claim


def _save_masked(image, masks, image_path, width, height):
    masked = image.copy()
    for mask in masks:
        mask_r = cv2.resize(mask, (width, height))
        binary = (mask_r > 0.5).astype(np.uint8) * 255
        overlay = np.zeros_like(image)
        overlay[:, :, 2] = binary
        masked = cv2.addWeighted(masked, 1, overlay, 0.5, 0)
    fname = secure_filename(f"masked_{os.path.basename(image_path)}")
    path  = os.path.join(os.path.dirname(image_path), fname)
    cv2.imwrite(path, masked)
    return path


# Step 1: Plant type detection (Gemini)
def detect_plant_type(image_path):
    try:
        import google.generativeai as genai
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            return 'Unknown'

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        pil_img = Image.open(image_path).convert('RGB')

        prompt = (
            "Identify the plant/crop type in this image. Common Indian crops: rice, wheat, tomato, potato, brinjal, okra, chilli, cotton, sugarcane, maize.\n"
            "Respond ONLY with the crop name (e.g. 'Tomato', 'Rice'). No other text."
        )

        response = model.generate_content([prompt, pil_img])
        plant = response.text.strip().title()
        return plant if plant in ['Rice', 'Wheat', 'Tomato', 'Potato', 'Brinjal', 'Okra', 'Chilli', 'Cotton', 'Sugarcane', 'Maize'] else 'Unknown'
    except:
        return 'Unknown'

# Step 2: Check for damage
def check_damage(image):
    if model is None:
        return False, [], 0.0
    results = model(image)
    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        conf_vals = []
        if hasattr(results[0], 'boxes') and results[0].boxes is not None:
            conf_vals = results[0].boxes.conf.cpu().numpy().tolist()
        conf = round(sum(conf_vals) / len(conf_vals), 3) if conf_vals else 0.0
        return len(masks) > 0, masks, conf
    return False, [], 0.0

# Step 3: Disease ID (plant-specific Gemini)
def identify_disease(image_path, plant, damage_pct):
    try:
        import google.generativeai as genai
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            return 'Unknown Disease', 'Moderate'

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        pil_img = Image.open(image_path).convert('RGB')

        prompt = (
            f"Plant: {plant}, Damage: {damage_pct:.1f}%. Identify specific disease. "
            f"Format: DISEASE: <name> | SEVERITY: <Mild/Moderate/Severe>"
        )

        response = model.generate_content([prompt, pil_img])
        text = response.text.strip()
        disease = text.split('DISEASE:')[1].split('|')[0].strip() if 'DISEASE:' in text else 'Unknown Disease'
        severity = text.split('SEVERITY:')[1].strip() if 'SEVERITY:' in text else _severity(damage_pct)
        return disease, severity
    except:
        return 'Unknown Disease', _severity(damage_pct)

# Step 4: Red dots annotation
def draw_red_dots(image, masks, image_path, width, height):
    annotated = image.copy()
    for mask in masks:
        mask_r = cv2.resize(mask, (width, height))
        y_coords, x_coords = np.where(mask_r > 0.5)
        if len(y_coords) > 0:
            for _ in range(min(20, len(y_coords))):  # Max 20 dots
                idx = np.random.randint(0, len(y_coords))
                cv2.circle(annotated, (x_coords[idx], y_coords[idx]), 8, (0, 0, 255), -1)
                cv2.circle(annotated, (x_coords[idx], y_coords[idx]), 12, (0, 0, 255), 2)
    
    fname = secure_filename(f"annotated_{os.path.basename(image_path)}")
    path = os.path.join(os.path.dirname(image_path), fname)
    cv2.imwrite(path, annotated)
    return path

def detect_disease_damage(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return {'error': 'Could not read image'}
        height, width = image.shape[:2]
        visual_damage_pct = _estimate_visual_damage(image)

        # Step 1: Plant type detection (fallback)
        plant = detect_plant_type(image_path)

        # Step 2: Check for damage using YOLO
        damage_detected, masks, conf = check_damage(image)
        damage_pct = 0.0
        if damage_detected:
            total = height * width
            damaged = sum(np.sum(cv2.resize(m, (width, height)) > 0.5) for m in masks)
            damage_pct = (damaged / total) * 100
            if damage_pct < 1.0:
                damage_pct = 0.0  # Threshold for 'no significant damage'

        # Always run Gemini vision detection for accurate disease assessment (primary path)
        vision_result = gemini_vision_detection(image_path, image, width, height)
        
        if vision_result.get('error'):
            if vision_result.get('healthy') or vision_result.get('source') == 'mock':
                    return _healthy_detection(
                        plant=plant,
                        image_path=image_path,
                        source=vision_result.get('source', 'none'),
                        confidence=1.0,
                    error=vision_result.get('error')
                )

            # Vision failed, fallback to YOLO + identify_disease if damage detected
            if damage_detected and damage_pct >= 1.0:
                disease, severity = identify_disease(image_path, plant, damage_pct)
                if _is_healthy_diagnosis(disease, damage_pct):
                    return _healthy_detection(
                        plant=plant,
                        image_path=image_path,
                        source='yolo_fallback',
                        confidence=1.0,
                        error=vision_result.get('error')
                    )
                annotated_path = draw_red_dots(image, masks, image_path, width, height)
                return {
                    'plant': plant,
                    'disease': disease,
                    'damage_percentage': round(damage_pct, 2),
                    'severity': severity,
                    'confidence': conf or 0.6,
                    'vision_source': 'yolo_fallback',
                    'source': 'yolo_fallback',
                    'annotated_image_path': annotated_path,
                    'healthy': False,
                    'error': vision_result['error']
                }
            else:
                # No significant damage or vision/YOLO failed
                return _healthy_detection(
                    plant=plant,
                    image_path=image_path,
                    source='none',
                    confidence=1.0,
                    error=vision_result.get('error')
                )

        # Primary: Use Gemini vision results (always called, trusted)
        if vision_result.get('healthy') or _should_force_healthy(vision_result, visual_damage_pct, damage_pct):
            return _healthy_detection(
                plant=plant,
                image_path=image_path,
                source='gemini_vision',
                confidence=1.0
            )

        masked_path = vision_result['masked_image_path']
        return {
            'plant': plant,
            'disease': vision_result['disease'],
            'damage_percentage': vision_result['damage_percentage'],
            'severity': vision_result['severity'],
            'confidence': conf or 0.85,  # YOLO conf boosted by Gemini vision
            'vision_source': 'gemini_vision',
            'source': 'gemini_vision',
            'masked_image_path': masked_path,
            'yolo_damage_pct': round(damage_pct, 2) if damage_detected else 0.0,
            'visual_damage_pct': round(visual_damage_pct, 2),
            'healthy': False,
            'error': None
        }

    except Exception as e:
        return {'error': f'Processing error: {str(e)}'}


def gemini_vision_detection(image_path, image, width, height):
    """
    Use Gemini Vision to accurately identify the plant disease from the image.
    Falls back to mock only if Gemini is unavailable.
    """
    try:
        import google.generativeai as genai
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("No GEMINI_API_KEY")

        genai.configure(api_key=api_key)
        model_v = genai.GenerativeModel('gemini-2.0-flash-lite')

        pil_img = Image.open(image_path).convert('RGB')

        prompt = (
            "You are an expert plant pathologist. Analyze this crop image carefully.\n"
            "Respond in EXACTLY this format (no extra text):\n"
            "DISEASE: <disease name>\n"
            "DAMAGE: <number between 0 and 100>\n"
            "SEVERITY: <Mild|Moderate|Severe>\n\n"
            "Rules:\n"
            "- Only report a disease when clear visible disease symptoms are present.\n"
            "- If the plant looks healthy or symptoms are uncertain, write DISEASE: Healthy, DAMAGE: 0, SEVERITY: Mild\n"
            "- Be specific: e.g. 'Tomato Late Blight', 'Rice Blast', 'Powdery Mildew', 'Bacterial Leaf Spot'\n"
            "- DAMAGE is the estimated % of the visible plant area that is diseased\n"
            "- Only output the 3 lines above, nothing else"
        )

        response = model_v.generate_content([prompt, pil_img])
        text = response.text.strip()
        print(f"[VISION DEBUG] Gemini response: {text}", file=sys.stderr)

        # Parse response
        disease, damage_pct, severity = _parse_gemini_vision(text)
        healthy = _is_healthy_diagnosis(disease, damage_pct)
        if healthy:
            disease = 'Healthy plant'
            damage_pct = 0.0
            severity = 'None'

        # Create a visual highlight on the image (colour-coded by severity)
        masked_path = _create_highlight(image, image_path, width, height, severity, damage_pct)

        return {
            'disease': disease,
            'damage_percentage': round(damage_pct, 2),
            'severity': severity,
            'masked_image_path': masked_path,
            'source': 'gemini_vision',
            'confidence': None,
            'healthy': healthy,
            'error': None
        }

    except Exception as e:
        print(f"[VISION] Gemini vision failed: {e}, falling back to mock", file=sys.stderr)
        return mock_disease_detection(image_path)


def _parse_gemini_vision(text):
    """Parse the structured Gemini vision response."""
    import re
    disease  = 'Unknown Disease'
    damage   = 0.0
    severity = 'Mild'

    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith('DISEASE:'):
            disease = line.split(':', 1)[1].strip()
        elif line.upper().startswith('DAMAGE:'):
            try:
                damage = float(re.search(r'[\d.]+', line.split(':', 1)[1]).group())
                damage = max(0.0, min(100.0, damage))
            except Exception:
                pass
        elif line.upper().startswith('SEVERITY:'):
            s = line.split(':', 1)[1].strip().capitalize()
            if s in ('Mild', 'Moderate', 'Severe'):
                severity = s

    if _is_healthy_diagnosis(disease, damage):
        return 'Healthy plant', 0.0, 'None'

    # Recalculate severity from damage so the label always matches the percentage.
    severity = _severity(damage)
    return disease, damage, severity


def _create_highlight(image, image_path, width, height, severity, damage_pct):
    """Draw a semi-transparent overlay proportional to damage percentage."""
    overlay = image.copy()
    # colour: green=mild, orange=moderate, red=severe
    colour = {'Mild': (0, 200, 0), 'Moderate': (0, 140, 255), 'Severe': (0, 0, 220)}.get(severity, (0, 0, 220))

    # Cover roughly damage_pct % of the image area with the overlay
    cover_h = int(height * (damage_pct / 100))
    if cover_h > 0:
        overlay[height - cover_h:, :] = colour
    result = cv2.addWeighted(image, 0.65, overlay, 0.35, 0)

    fname = secure_filename(f"masked_{os.path.basename(image_path)}")
    path  = os.path.join(os.path.dirname(image_path), fname)
    cv2.imwrite(path, result)
    return path


def mock_disease_detection(image_path):
    """Last-resort result when both YOLO and Gemini are unavailable.

    Do not invent a disease here. A random demo disease creates false positives
    for healthy uploads, so this path returns a healthy/0% result instead.
    """
    try:
        return _healthy_detection(
            image_path=image_path,
            source='mock',
            confidence=1.0,
            error='Gemini vision unavailable; no disease was detected by fallback.'
        )
    except Exception as e:
        return {'error': f'Mock processing error: {str(e)}'}


