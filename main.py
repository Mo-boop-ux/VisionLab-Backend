import time
import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import cv2
import numpy as np
import base64
import requests

app = FastAPI(title="VisionLab Advanced Computer Vision API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Haar Cascades from local models directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
face_cascade_path = os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml")
eye_cascade_path = os.path.join(MODELS_DIR, "haarcascade_eye.xml")

face_cascade = cv2.CascadeClassifier(face_cascade_path) if os.path.exists(face_cascade_path) else None
eye_cascade = cv2.CascadeClassifier(eye_cascade_path) if os.path.exists(eye_cascade_path) else None

class ProcessRequest(BaseModel):
    image_uri: str
    operation: str
    threshold: int = 128
    blur_kernel: int = 7
    brightness: int = 0
    contrast: float = 1.0
    channel: str = "red"
    param_1: float = 35.0  # General parameter (min_area for counter, cutoff radius for FFT, etc.)
    mode: str = "default"  # Sub-mode (e.g. warp, binarize, corners)

def decode_data_or_url(value: str):
    if value.startswith("data:") and "," in value:
        encoded = value.split(",", 1)[1]
        data = base64.b64decode(encoded)
    elif not value.startswith("http://") and not value.startswith("https://"):
        data = base64.b64decode(value)
    else:
        r = requests.get(value, timeout=20)
        r.raise_for_status()
        data = r.content

    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")
    return image

def encode_image(image):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError("Could not encode image")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode()

def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 (x,y) points in clockwise order: Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]       # Top-Left has smallest sum
    rect[2] = pts[np.argmax(s)]       # Bottom-Right has largest sum
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]    # Top-Right has smallest difference
    rect[3] = pts[np.argmax(diff)]    # Bottom-Left has largest difference
    return rect

def find_document_contour(img: np.ndarray):
    """Detects the largest 4-point convex polygon in the image for document scanning."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 60, 180)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:6]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > (img.shape[0] * img.shape[1] * 0.05):
            return approx.reshape(4, 2)
    return None

def run_document_scanner(img: np.ndarray, mode: str):
    """Applies Homography / Perspective Warp to straighten documents or highlights detected corners."""
    h, w = img.shape[:2]
    doc_pts = find_document_contour(img)

    if doc_pts is None:
        # Fallback quad (slight inset) if no sharp 4-corner document detected
        doc_pts = np.array([
            [int(w * 0.08), int(h * 0.08)],
            [int(w * 0.92), int(h * 0.08)],
            [int(w * 0.92), int(h * 0.92)],
            [int(w * 0.08), int(h * 0.92)],
        ], dtype="float32")
        detected = False
    else:
        detected = True

    ordered_pts = order_quad_points(doc_pts)
    tl, tr, br, bl = ordered_pts

    # Compute width and height of new perspective corrected image
    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    max_w = max(int(w_top), int(w_bot), 100)

    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)
    max_h = max(int(h_left), int(h_right), 100)

    dst_pts = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (max_w, max_h))

    cv_stats = {
        "doc_detected": detected,
        "corners": [[int(p[0]), int(p[1])] for p in ordered_pts],
        "aspect_ratio": round(float(max_w / max_h), 2),
        "warped_width": max_w,
        "warped_height": max_h,
    }

    if mode == "corners":
        # Draw detected document polygon and glowing corner points on original
        annotated = img.copy()
        pts_int = ordered_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts_int], True, (0, 255, 230), 3, cv2.LINE_AA)
        corner_names = ["TL", "TR", "BR", "BL"]
        for idx, (pt, name) in enumerate(zip(ordered_pts, corner_names)):
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(annotated, (x, y), 9, (255, 0, 128), -1)
            cv2.circle(annotated, (x, y), 13, (0, 255, 230), 2)
            cv2.putText(annotated, f"{name}:({x},{y})", (x + 12, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"{name}:({x},{y})", (x + 12, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return annotated, cv_stats

    elif mode == "binarize":
        # Straightened + high-contrast adaptive binarization for document text clarity
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        binarized = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
        )
        return cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR), cv_stats

    else:
        # Default perspective warped scan in full color
        return warped, cv_stats

def run_object_counter(img: np.ndarray, min_area: float = 35.0):
    """Quantitative Particle / Cell / Object Counter using Connected Components & Morphology."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu automatic thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Check if background was white (standard microscopy) or dark
    white_pixels = np.sum(thresh == 255)
    total_pixels = thresh.size
    if white_pixels > 0.65 * total_pixels:
        thresh = cv2.bitwise_not(thresh)

    # Morphological opening to separate touching cells
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Connected Components with Statistics
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

    annotated = img.copy()
    valid_objects = []
    total_area_occupied = 0

    min_sz = max(10, int(min_area))
    obj_index = 1

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_sz:
            continue
        
        total_area_occupied += area
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        # Radius for bounding circle
        radius = max(8, int(np.sqrt(area / np.pi) * 1.25))

        # Color palette cycle
        colors = [(52, 211, 153), (99, 102, 241), (244, 63, 94), (245, 158, 11), (14, 165, 233)]
        color = colors[(obj_index - 1) % len(colors)]

        # Draw target ring and index
        cv2.circle(annotated, (cx, cy), radius, color, 2, cv2.LINE_AA)
        cv2.circle(annotated, (cx, cy), 3, (255, 255, 255), -1)

        # Label background pill
        label_text = f"#{obj_index}"
        cv2.putText(annotated, label_text, (cx - 10, cy - radius - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated, label_text, (cx - 10, cy - radius - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        valid_objects.append({
            "id": obj_index,
            "centroid": [cx, cy],
            "area_px": area,
        })
        obj_index += 1

    count = len(valid_objects)
    areas = [o["area_px"] for o in valid_objects]
    mean_area = round(float(np.mean(areas)), 1) if count > 0 else 0
    coverage_pct = round(float((total_area_occupied / total_pixels) * 100), 2)

    cv_stats = {
        "object_count": count,
        "mean_area_px": mean_area,
        "max_area_px": max(areas) if count > 0 else 0,
        "min_area_px": min(areas) if count > 0 else 0,
        "density_coverage": coverage_pct,
    }
    return annotated, cv_stats

def run_face_biometrics(img: np.ndarray):
    """Haar Cascade Face & Eye Biometric Tracker with Cyberpunk HUD Overlay."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    annotated = img.copy()

    faces = []
    if face_cascade and not face_cascade.empty():
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(36, 36))

    eyes_count = 0
    face_boxes = []

    for idx, (x, y, w, h) in enumerate(faces):
        face_boxes.append([int(x), int(y), int(w), int(h)])
        
        # Cyberpunk corner brackets on face
        line_len = int(min(w, h) * 0.22)
        color = (0, 240, 255) # Neon cyan
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 120, 160), 1)

        # Top-Left bracket
        cv2.line(annotated, (x, y), (x + line_len, y), color, 3)
        cv2.line(annotated, (x, y), (x, y + line_len), color, 3)
        # Top-Right bracket
        cv2.line(annotated, (x + w, y), (x + w - line_len, y), color, 3)
        cv2.line(annotated, (x + w, y), (x + w, y + line_len), color, 3)
        # Bottom-Left bracket
        cv2.line(annotated, (x, y + h), (x + line_len, y + h), color, 3)
        cv2.line(annotated, (x, y + h), (x, y + h - line_len), color, 3)
        # Bottom-Right bracket
        cv2.line(annotated, (x + w, y + h), (x + w - line_len, y + h), color, 3)
        cv2.line(annotated, (x + w, y + h), (x + w, y + h - line_len), color, 3)

        # Face Label
        label = f"SUBJECT #{idx + 1} ({w}x{h})"
        cv2.putText(annotated, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Detect eyes within face ROI
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = annotated[y:y + h, x:x + w]
        if eye_cascade and not eye_cascade.empty():
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.15, minNeighbors=3, minSize=(14, 14))
            for (ex, ey, ew, eh) in eyes:
                eyes_count += 1
                ecx, ecy = ex + ew // 2, ey + eh // 2
                cv2.circle(roi_color, (ecx, ecy), max(6, ew // 2), (255, 0, 180), 2)
                cv2.circle(roi_color, (ecx, ecy), 2, (255, 255, 255), -1)

    cv_stats = {
        "faces_detected": len(faces),
        "eyes_detected": eyes_count,
        "face_boxes": face_boxes,
    }
    return annotated, cv_stats

def run_feature_keypoints(img: np.ndarray, method: str = "orb"):
    """Keypoint & Feature extraction using Harris Corner Detection or ORB (FAST+BRIEF)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    annotated = img.copy()

    if method == "harris":
        # Harris Corner Response
        harris = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        harris_dilated = cv2.dilate(harris, None)
        threshold_val = 0.01 * harris_dilated.max()
        corners = np.argwhere(harris_dilated > threshold_val)

        # Draw corner crosshairs
        for (y, x) in corners[:600]:
            cv2.circle(annotated, (int(x), int(y)), 3, (0, 255, 120), -1)
            cv2.drawMarker(annotated, (int(x), int(y)), (255, 0, 180), cv2.MARKER_CROSS, 7, 1)

        cv_stats = {
            "keypoints_count": len(corners),
            "method": "Harris Corner Response (2nd Moment Matrix)",
        }
        return annotated, cv_stats

    else: # ORB
        orb = cv2.ORB_create(nfeatures=400)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        annotated = cv2.drawKeypoints(
            img,
            keypoints,
            None,
            color=(0, 255, 200),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        cv_stats = {
            "keypoints_count": len(keypoints),
            "method": "ORB (Oriented FAST & Rotated BRIEF)",
            "descriptor_dims": descriptors.shape if descriptors is not None else [0, 0],
        }
        return annotated, cv_stats

def run_fourier_fft(img: np.ndarray, mode: str = "spectrum", cutoff: float = 35.0):
    """2D Discrete Fourier Transform (FFT) analysis and Frequency Domain Filtering."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    # Compute 2D FFT & shift DC component to center
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)

    radius = max(5, int(cutoff))

    if mode == "spectrum":
        # Magnitude spectrum: 20 * log(|F| + 1)
        magnitude_spectrum = 20 * np.log(mag + 1)
        # Normalize to 0-255
        norm_mag = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
        norm_mag = np.uint8(norm_mag)
        # Apply scientific colormap
        colored = cv2.applyColorMap(norm_mag, cv2.COLORMAP_MAGMA)
        
        cv_stats = {
            "fft_dimensions": f"{cols}x{rows}",
            "dc_frequency_intensity": round(float(np.max(magnitude_spectrum)), 1),
            "domain": "2D Spatial Frequency Domain (Fourier Spectrum)",
        }
        return colored, cv_stats

    elif mode == "lowpass":
        # Low-Pass Filter: Zero out high frequencies beyond cutoff radius
        y, x = np.ogrid[:rows, :cols]
        mask = ((x - ccol) ** 2 + (y - crow) ** 2) <= (radius ** 2)
        fshift_filtered = fshift * mask
        
        # Inverse 2D FFT back to spatial domain
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        img_back = np.clip(img_back, 0, 255).astype(np.uint8)
        
        cv_stats = {
            "filter_type": "Ideal Low-Pass Frequency Filter",
            "cutoff_radius_px": radius,
            "domain": "Inverse FFT (Frequency Smoothing)",
        }
        return cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR), cv_stats

    elif mode == "highpass":
        # High-Pass Filter: Attenuate low frequencies inside cutoff radius
        y, x = np.ogrid[:rows, :cols]
        mask = ((x - ccol) ** 2 + (y - crow) ** 2) > (radius ** 2)
        fshift_filtered = fshift * mask
        
        # Inverse 2D FFT back to spatial domain
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        img_back = np.clip(img_back, 0, 255).astype(np.uint8)
        
        cv_stats = {
            "filter_type": "Ideal High-Pass Frequency Filter",
            "cutoff_radius_px": radius,
            "domain": "Inverse FFT (High-Frequency Edge Reconstruction)",
        }
        return cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR), cv_stats

def run_clahe_enhancement(img: np.ndarray, clip_limit: float = 3.0):
    """Contrast Limited Adaptive Histogram Equalization on LAB Luminance."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=max(1.0, float(clip_limit)), tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    
    merged = cv2.merge((l_eq, a, b))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    # Compute 64-bin luminance histogram for mobile chart
    hist = cv2.calcHist([l_eq], [0], None, [64], [0, 256]).flatten()
    hist_norm = [round(float(v / hist.max()), 3) for v in hist]

    cv_stats = {
        "clip_limit": clip_limit,
        "tile_grid": "8x8",
        "histogram_bins": hist_norm,
    }
    return result, cv_stats

def run_color_segmentation(img: np.ndarray, color_target: str = "red"):
    """HSV Color Space Segmentation & Bounding Box Localization."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    if color_target == "red":
        lower1 = np.array([0, 70, 50])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 70, 50])
        upper2 = np.array([180, 255, 255])
        mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
    elif color_target == "green":
        lower = np.array([35, 60, 40])
        upper = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    elif color_target == "blue":
        lower = np.array([95, 60, 40])
        upper = np.array([135, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    elif color_target == "skin":
        lower = np.array([0, 30, 60])
        upper = np.array([20, 150, 255])
        mask = cv2.inRange(hsv, lower, upper)
    else:
        mask = np.full(img.shape[:2], 255, dtype=np.uint8)

    # Clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Apply mask on color image
    segmented = cv2.bitwise_and(img, img, mask=mask)
    
    # Locate largest matching contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated = segmented.copy()
    count = 0
    for c in contours:
        if cv2.contourArea(c) > 150:
            count += 1
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv_stats = {
        "color_target": color_target,
        "segments_found": count,
        "mask_coverage": round(float(np.sum(mask > 0) / mask.size * 100), 2),
    }
    return annotated, cv_stats

def run_operation(
    img: np.ndarray,
    op: str,
    threshold: int = 128,
    blur_kernel: int = 7,
    brightness: int = 0,
    contrast: float = 1.0,
    channel: str = "red",
    param_1: float = 35.0,
    mode: str = "default",
):
    """Dispatches requested operation to appropriate OpenCV CV algorithm."""
    k = max(3, int(blur_kernel))
    if k % 2 == 0:
        k += 1

    # --- ADVANCED COMPUTER VISION SUITE ---
    if op == "doc_scan":
        result, extra_stats = run_document_scanner(img, mode="warp")
        return result, extra_stats

    elif op == "doc_corners":
        result, extra_stats = run_document_scanner(img, mode="corners")
        return result, extra_stats

    elif op == "doc_binarize":
        result, extra_stats = run_document_scanner(img, mode="binarize")
        return result, extra_stats

    elif op == "object_counter":
        result, extra_stats = run_object_counter(img, min_area=param_1)
        return result, extra_stats

    elif op == "face_detect":
        result, extra_stats = run_face_biometrics(img)
        return result, extra_stats

    elif op == "harris_corners":
        result, extra_stats = run_feature_keypoints(img, method="harris")
        return result, extra_stats

    elif op == "orb_features":
        result, extra_stats = run_feature_keypoints(img, method="orb")
        return result, extra_stats

    elif op == "fft_spectrum":
        result, extra_stats = run_fourier_fft(img, mode="spectrum")
        return result, extra_stats

    elif op == "fft_lowpass":
        result, extra_stats = run_fourier_fft(img, mode="lowpass", cutoff=param_1)
        return result, extra_stats

    elif op == "fft_highpass":
        result, extra_stats = run_fourier_fft(img, mode="highpass", cutoff=param_1)
        return result, extra_stats

    elif op == "clahe":
        result, extra_stats = run_clahe_enhancement(img, clip_limit=contrast)
        return result, extra_stats

    elif op == "color_segment":
        result, extra_stats = run_color_segmentation(img, color_target=channel)
        return result, extra_stats

    # --- STANDARD PROCESSING KERNELS ---
    elif op == "grayscale":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), {}

    elif op == "negative":
        return 255 - img, {}

    elif op == "sepia":
        sepia_matrix = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        transformed = cv2.transform(img, sepia_matrix)
        return np.clip(transformed, 0, 255).astype(np.uint8), {}

    elif op == "threshold":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), {}

    elif op == "otsu":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        val, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR), {"otsu_threshold": int(val)}

    elif op == "blur":
        return cv2.GaussianBlur(img, (k, k), 0), {}

    elif op == "median_blur":
        return cv2.medianBlur(img, k), {}

    elif op == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(img, -1, kernel), {}

    elif op == "emboss":
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        embossed = cv2.filter2D(gray, -1, kernel) + 128
        embossed = np.clip(embossed, 0, 255).astype(np.uint8)
        return cv2.cvtColor(embossed, cv2.COLOR_GRAY2BGR), {}

    elif op == "edges":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 80, 180)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), {}

    elif op == "sobel":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.hypot(sobelx, sobely)
        mag = (mag / (mag.max() + 1e-6) * 255).astype(np.uint8)
        return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR), {}

    elif op == "laplacian":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap = np.uint8(np.clip(np.absolute(lap), 0, 255))
        return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR), {}

    elif op == "equalize":
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR), {}

    elif op == "brightness_contrast":
        alpha = max(0.2, min(3.0, float(contrast)))
        beta = max(-100, min(100, int(brightness)))
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta), {}

    elif op == "cartoon":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 7)
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        color = cv2.bilateralFilter(img, d=9, sigmaColor=150, sigmaSpace=150)
        return cv2.bitwise_and(color, edges_color), {}

    elif op in ["dilate", "erode", "morph_open", "morph_close"]:
        elem = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        if op == "dilate":
            res = cv2.dilate(img, elem, iterations=1)
        elif op == "erode":
            res = cv2.erode(img, elem, iterations=1)
        elif op == "morph_open":
            res = cv2.morphologyEx(img, cv2.MORPH_OPEN, elem)
        else: # morph_close
            res = cv2.morphologyEx(img, cv2.MORPH_CLOSE, elem)
        return res, {}

    elif op == "contours":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = img.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 120), 2)
        return result, {"contours_count": len(contours)}

    elif op == "channel":
        result = np.zeros_like(img)
        if channel == "blue":
            result[:, :, 0] = img[:, :, 0]
        elif channel == "green":
            result[:, :, 1] = img[:, :, 1]
        else:
            result[:, :, 2] = img[:, :, 2]
        return result, {}

    else:
        raise HTTPException(status_code=400, detail=f"Unknown operation: {op}")

def compute_stats(img: np.ndarray, duration_ms: float, extra: Optional[Dict[str, Any]] = None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) > 2 else img
    base_stats = {
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "channels": int(img.shape[2]) if len(img.shape) > 2 else 1,
        "mean_intensity": round(float(np.mean(gray)), 1),
        "execution_ms": round(duration_ms, 1),
    }
    if extra:
        base_stats.update(extra)
    return base_stats

def generate_sample(sample_id: str) -> np.ndarray:
    """Generates synthetic computer vision test benchmark scenes."""
    if sample_id == "cells":
        img = np.full((500, 500, 3), 242, dtype=np.uint8)
        # Background vignette
        cv2.circle(img, (250, 250), 240, (230, 230, 235), -1)
        rng = np.random.RandomState(42)
        for _ in range(24):
            cx = int(rng.randint(50, 450))
            cy = int(rng.randint(50, 450))
            r = int(rng.randint(12, 36))
            color = (int(rng.randint(40, 90)), int(rng.randint(30, 70)), int(rng.randint(90, 180)))
            cv2.circle(img, (cx, cy), r, color, -1)
            cv2.circle(img, (cx, cy), max(2, r // 3), (255, 255, 255), -1)
            # Add subtle outer cell membrane
            cv2.circle(img, (cx, cy), r + 2, (120, 100, 140), 1)
        return img

    elif sample_id == "document":
        img = np.full((600, 600, 3), (40, 35, 30), dtype=np.uint8)
        # Add wood desk grain lines
        for y in range(0, 600, 25):
            cv2.line(img, (0, y), (600, y + 10), (48, 42, 36), 1)
        # Slanted document quad
        pts = np.array([[110, 75], [525, 125], [465, 535], [65, 445]], dtype=np.int32)
        cv2.fillPoly(img, [pts], (248, 248, 248))
        cv2.polylines(img, [pts], True, (160, 160, 160), 2, cv2.LINE_AA)
        # Draw mock document text lines inside polygon
        for line_y in range(160, 420, 28):
            p1 = (130 + (line_y - 160) // 4, line_y)
            p2 = (440 + (line_y - 160) // 4, line_y + 35)
            cv2.line(img, p1, p2, (70, 70, 70), 5, cv2.LINE_AA)
        # Header banner on page
        cv2.line(img, (140, 125), (460, 155), (30, 80, 210), 10, cv2.LINE_AA)
        return img

    elif sample_id == "checkerboard":
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        sq = 50
        for r in range(10):
            for c in range(10):
                if (r + c) % 2 == 0:
                    cv2.rectangle(img, (c * sq, r * sq), ((c + 1) * sq, (r + 1) * sq), (255, 255, 255), -1)
        # Add central calibration circle & concentric targets
        cv2.circle(img, (250, 250), 120, (0, 200, 255), 4)
        cv2.circle(img, (250, 250), 60, (255, 0, 180), 3)
        cv2.drawMarker(img, (250, 250), (0, 255, 100), cv2.MARKER_CROSS, 40, 2)
        return img

    elif sample_id == "wave":
        x = np.linspace(0, 12 * np.pi, 500)
        y = np.linspace(0, 12 * np.pi, 500)
        X, Y = np.meshgrid(x, y)
        Z = 128 + 55 * np.sin(X * 1.4) + 55 * np.cos(Y * 2.1) + 20 * np.sin((X + Y) * 0.8)
        Z = np.clip(Z, 0, 255).astype(np.uint8)
        return cv2.cvtColor(Z, cv2.COLOR_GRAY2BGR)

    elif sample_id == "colors":
        img = np.full((500, 500, 3), 40, dtype=np.uint8)
        # Red circle
        cv2.circle(img, (150, 150), 75, (30, 30, 220), -1)
        cv2.putText(img, "RED", (125, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Green rectangle
        cv2.rectangle(img, (280, 80), (430, 230), (30, 210, 30), -1)
        cv2.putText(img, "GREEN", (315, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Blue triangle
        tri_pts = np.array([[250, 280], [140, 440], [360, 440]], dtype=np.int32)
        cv2.fillPoly(img, [tri_pts], (220, 80, 20))
        cv2.putText(img, "BLUE", (220, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img

    else:
        # Default gradient test pattern
        base = np.zeros((400, 400, 3), dtype=np.uint8)
        for i in range(400):
            base[i, :] = [int(i * 255 / 400), int(128), int(255 - i * 255 / 400)]
        return base

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "3.0.0",
        "face_cascade_loaded": face_cascade is not None and not face_cascade.empty(),
        "eye_cascade_loaded": eye_cascade is not None and not eye_cascade.empty(),
    }

@app.get("/samples")
def list_samples():
    return [
        {
            "id": "cells",
            "title": "Cell / Particle Culture",
            "icon": "radio-button-on-outline",
            "desc": "Synthetic Petri dish with 24 distinct particles. Optimized for Particle Counter & Otsu.",
            "recommended": "object_counter",
        },
        {
            "id": "document",
            "title": "Slanted Document Page",
            "icon": "document-text-outline",
            "desc": "Perspective-distorted invoice on desk. Optimized for Document Warp & Clean Text.",
            "recommended": "doc_scan",
        },
        {
            "id": "checkerboard",
            "title": "Corner & Feature Target",
            "icon": "grid-outline",
            "desc": "High-contrast calibration grid. Optimized for Harris Corners & ORB Keypoints.",
            "recommended": "harris_corners",
        },
        {
            "id": "wave",
            "title": "Spatial Wave Harmonics",
            "icon": "pulse-outline",
            "desc": "2D sinusoidal interference wave. Optimized for 2D FFT Spectrum & Frequency Filters.",
            "recommended": "fft_spectrum",
        },
        {
            "id": "colors",
            "title": "Multicolor Calibration",
            "icon": "color-palette-outline",
            "desc": "Pure primary color primitives. Optimized for HSV Segmentation & Channel extraction.",
            "recommended": "color_segment",
        },
    ]

@app.get("/sample/{sample_id}")
def get_sample(sample_id: str):
    img = generate_sample(sample_id)
    return {
        "id": sample_id,
        "image_base64": encode_image(img),
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
    }

@app.post("/process")
def process(req: ProcessRequest):
    try:
        t0 = time.perf_counter()
        img = decode_data_or_url(req.image_uri)
        result, extra_stats = run_operation(
            img=img,
            op=req.operation,
            threshold=req.threshold,
            blur_kernel=req.blur_kernel,
            brightness=req.brightness,
            contrast=req.contrast,
            channel=req.channel,
            param_1=req.param_1,
            mode=req.mode,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            "operation": req.operation,
            "image_base64": encode_image(result),
            "stats": compute_stats(result, elapsed_ms, extra_stats),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/process-upload")
async def process_upload(
    file: UploadFile = File(...),
    operation: str = Form(...),
    threshold: int = Form(128),
    blur_kernel: int = Form(7),
    brightness: int = Form(0),
    contrast: float = Form(1.0),
    channel: str = Form("red"),
    param_1: float = Form(35.0),
    mode: str = Form("default"),
):
    try:
        t0 = time.perf_counter()
        data = await file.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        result, extra_stats = run_operation(
            img=img,
            op=operation,
            threshold=threshold,
            blur_kernel=blur_kernel,
            brightness=brightness,
            contrast=contrast,
            channel=channel,
            param_1=param_1,
            mode=mode,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            "operation": operation,
            "image_base64": encode_image(result),
            "stats": compute_stats(result, elapsed_ms, extra_stats),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
