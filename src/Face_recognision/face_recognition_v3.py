# face_recognition_module.py
import cv2
import numpy as np
from deepface import DeepFace
import tkinter as tk
from tkinter import filedialog
import json
import os
import time

# --- Configuration ---
CHOSEN_MODEL_NAME = "ArcFace"
DEFAULT_DETECTOR_BACKEND = "opencv"
DEFAULT_DISTANCE_METRIC = "cosine"
MAX_IMAGE_DIMENSION = 640  # Reduced from 800 to 640 for faster processing

# --- Global placeholders for loaded models ---
_loaded_recognition_model_obj = None
_detector_primed = False

def warm_up_deepface_systems(model_name_to_load, detector_backend_to_load):
    """Pre-load models to reduce per-call overhead"""
    global _loaded_recognition_model_obj, _detector_primed
    
    print(f"Warming up DeepFace: Loading '{model_name_to_load}' model and '{detector_backend_to_load}' detector...")
    warmup_start_time = time.time()

    # Load recognition model
    if _loaded_recognition_model_obj is None:
        print(f"  Loading recognition model: '{model_name_to_load}'...")
        model_load_start = time.time()
        try:
            _loaded_recognition_model_obj = DeepFace.build_model(model_name_to_load)
            print(f"  Recognition model loaded in {time.time() - model_load_start:.2f}s.")
        except Exception as e:
            print(f"  CRITICAL ERROR loading model: {e}")
            raise

    # Prime face detector
    if not _detector_primed:
        print(f"  Priming detector: '{detector_backend_to_load}'...")
        detector_load_start = time.time()
        try:
            dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)
            _ = DeepFace.extract_faces(
                img_path=dummy_image,
                detector_backend=detector_backend_to_load,
                enforce_detection=False,
                align=False
            )
            _detector_primed = True
            print(f"  Detector primed in {time.time() - detector_load_start:.2f}s.")
        except Exception as e:
            print(f"  WARNING priming detector: {e}")

    total_warmup_time = time.time() - warmup_start_time
    print(f"--- Warm-up completed in {total_warmup_time:.2f} seconds ---")
    return total_warmup_time

def load_and_resize_image(image_path, max_dimension):
    """Load and resize image to optimize processing speed"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    orig_h, orig_w = img.shape[:2]
    if max(orig_h, orig_w) <= max_dimension:
        return img, (orig_h, orig_w), (orig_h, orig_w)
    
    scale = max_dimension / max(orig_h, orig_w)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_img, (orig_h, orig_w), (new_h, new_w)

def scale_bbox(bbox, orig_dims, resized_dims):
    """Convert face coordinates from resized image to original dimensions"""
    if not bbox:
        return None
    orig_h, orig_w = orig_dims
    resized_h, resized_w = resized_dims
    scale_x = orig_w / resized_w
    scale_y = orig_h / resized_h
    return {
        "x": int(bbox["x"] * scale_x),
        "y": int(bbox["y"] * scale_y),
        "w": int(bbox["w"] * scale_x),
        "h": int(bbox["h"] * scale_y)
    }

def verify_faces_single_model(selfie_path, id_photo_path,
                             model_name=CHOSEN_MODEL_NAME,
                             detector_backend=DEFAULT_DETECTOR_BACKEND,
                             distance_metric=DEFAULT_DISTANCE_METRIC,
                             custom_threshold=None):
    """Optimized face verification with image preprocessing"""
    results = {}
    try:
        # Validate paths
        if not os.path.exists(selfie_path):
            return {"error": f"Selfie not found: {selfie_path}"}
        if not os.path.exists(id_photo_path):
            return {"error": f"ID photo not found: {id_photo_path}"}

        # Preprocess images
        preprocess_start = time.time()
        selfie_img, selfie_orig_dims, selfie_resized_dims = load_and_resize_image(
            selfie_path, MAX_IMAGE_DIMENSION
        )
        id_img, id_orig_dims, id_resized_dims = load_and_resize_image(
            id_photo_path, MAX_IMAGE_DIMENSION
        )
        preprocess_time = time.time() - preprocess_start

        # Perform verification
        verify_start = time.time()
        result_obj = DeepFace.verify(
            img1_path=selfie_img,
            img2_path=id_img,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            enforce_detection=True,
            normalization="base",  # Skip normalization for faster processing
            align=True  # Alignment is necessary for accuracy
        )
        verify_time = time.time() - verify_start

        # Process results
        verified = result_obj["verified"]
        distance = float(result_obj["distance"])
        threshold = custom_threshold or float(result_obj["threshold"])
        
        if custom_threshold is not None:
            verified = distance <= custom_threshold

        # Scale face coordinates to original image size
        facial_areas = result_obj.get("facial_areas", {})
        img1_area = scale_bbox(
            facial_areas.get("img1"), 
            selfie_orig_dims, 
            selfie_resized_dims
        )
        img2_area = scale_bbox(
            facial_areas.get("img2"), 
            id_orig_dims, 
            id_resized_dims
        )

        results = {
            "verified": verified,
            "distance": distance,
            "threshold_used": threshold,
            "model": model_name,
            "detector": detector_backend,
            "preprocess_time": round(preprocess_time, 3),
            "verify_time": round(verify_time, 3),
            "total_time": round(preprocess_time + verify_time, 3),
            "facial_areas": {"img1": img1_area, "img2": img2_area}
        }

    except ValueError as ve:
        results = {"error": str(ve)}
    except Exception as e:
        results = {"error": f"Unexpected error: {str(e)}"}
    return results

def select_files_gui():
    """GUI for file selection without full Tk window"""
    root = tk.Tk()
    root.withdraw()
    print("Select selfie photo...")
    selfie_path = filedialog.askopenfilename(
        title="Select Selfie",
        filetypes=[("Images", "*.jpg *.jpeg *.png")]
    )
    if not selfie_path:
        return None, None
        
    print("Select ID photo...")
    id_path = filedialog.askopenfilename(
        title="Select ID Document",
        filetypes=[("Images", "*.jpg *.jpeg *.png")]
    )
    return selfie_path, id_path

if __name__ == "__main__":
    # Initialization
    start_time = time.time()
    print("=" * 60)
    print(f"INITIALIZING (Loading {CHOSEN_MODEL_NAME} model)")
    
    try:
        warmup_time = warm_up_deepface_systems(CHOSEN_MODEL_NAME, DEFAULT_DETECTOR_BACKEND)
    except Exception as e:
        print(f"Fatal error during warm-up: {e}")
        exit()

    # File selection
    print(f"\nSYSTEM READY. Warm-up: {warmup_time:.2f}s")
    print("=" * 60)
    selfie_path, id_path = select_files_gui()
    
    if not selfie_path or not id_path:
        print("File selection cancelled")
        exit()

    # Verification
    print(f"\nVerifying:\n- Selfie: {os.path.basename(selfie_path)}\n- ID: {os.path.basename(id_path)}")
    result = verify_faces_single_model(selfie_path, id_path)
    
    print("\n=== RESULTS ===")
    print(json.dumps(result, indent=2))
    
    if "error" not in result:
        status = "VERIFIED" if result["verified"] else "NOT VERIFIED"
        print(f"\nRESULT: {status} (Distance: {result['distance']:.4f}, Threshold: {result['threshold_used']})")
        print(f"TIMING: Preprocess: {result['preprocess_time']}s, Verify: {result['verify_time']}s, Total: {result['total_time']}s")
        
        if result['total_time'] < 2.0:
            print("✅ GOAL ACHIEVED: Total processing under 2 seconds!")
        else:
            print("⚠️ Total time over 2 seconds. Try reducing MAX_IMAGE_DIMENSION to 480")
    else:
        print(f"❌ ERROR: {result['error']}")

    print(f"\nTotal execution: {time.time() - start_time:.2f} seconds")
