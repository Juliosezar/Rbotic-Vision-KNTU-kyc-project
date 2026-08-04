# face_recognition_module.py
import cv2
import numpy as np
from deepface import DeepFace
import tkinter as tk
from tkinter import filedialog
import json
import os

# --- Configuration ---
# Choose the primary model you want to use.
# Options: "ArcFace", "Facenet512", "SFace", "VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "GhostFaceNet"
# ArcFace is generally a strong choice for accuracy and robustness.
CHOSEN_MODEL_NAME = "ArcFace"
DEFAULT_DETECTOR_BACKEND = 'retinaface' # Other options: 'opencv', 'ssd', 'dlib', 'mtcnn', 'yolov8'
DEFAULT_DISTANCE_METRIC = 'cosine' # 'cosine', 'euclidean', 'euclidean_l2'

def verify_faces_single_model(selfie_path, id_photo_path,
                              model_name=CHOSEN_MODEL_NAME,
                              detector_backend=DEFAULT_DETECTOR_BACKEND,
                              distance_metric=DEFAULT_DISTANCE_METRIC,
                              custom_threshold=None): # If None, uses model's default
    """
    Verify if faces in two images match using a single specified DeepFace model.

    Args:
        selfie_path (str): Path to selfie image.
        id_photo_path (str): Path to ID document photo.
        model_name (str): DeepFace model to use.
        detector_backend (str): Face detector backend.
        distance_metric (str): Metric for distance calculation.
        custom_threshold (float, optional): Override default threshold for the model.

    Returns:
        dict: Verification results.
    """
    results = {}
    try:
        if not os.path.exists(selfie_path):
            return {"error": f"Selfie image not found: {selfie_path}"}
        if not os.path.exists(id_photo_path):
            return {"error": f"ID photo not found: {id_photo_path}"}

        result_obj = DeepFace.verify(
            img1_path=selfie_path,
            img2_path=id_photo_path,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            enforce_detection=True # Recommended to ensure a face is found
        )

        verified_status = result_obj["verified"]
        threshold_used = result_obj["threshold"] # Model's default or threshold used by verify

        if custom_threshold is not None:
            # Override verification based on custom threshold
            if result_obj["distance"] <= custom_threshold:
                verified_status = True
            else:
                verified_status = False
            threshold_used = custom_threshold # Reflect the threshold we actually used for decision

        # Ensure all numerical values are Python native types for JSON serialization
        distance = result_obj["distance"]
        if isinstance(distance, (np.float32, np.float64)):
            distance = float(distance)
        if isinstance(threshold_used, (np.float32, np.float64)):
            threshold_used = float(threshold_used)
        if isinstance(verified_status, np.bool_):
            verified_status = bool(verified_status)
        
        model_default_thresh = result_obj.get("threshold", "N/A") # Get original threshold
        if isinstance(model_default_thresh, (np.float32, np.float64)):
            model_default_thresh = float(model_default_thresh)


        results = {
            "verified": verified_status,
            "distance": distance,
            "threshold_used_for_decision": threshold_used,
            "model_default_threshold": model_default_thresh,
            "model": model_name,
            "detector_backend": detector_backend,
            "distance_metric": distance_metric,
            "facial_areas": {
                "img1": result_obj.get("facial_areas", {}).get("img1"),
                "img2": result_obj.get("facial_areas", {}).get("img2")
            },
            "time_seconds": result_obj.get("time")
        }

    except ValueError as ve: # Often "Face could not be detected."
        results = {"error": str(ve), "model": model_name, "detector_backend": detector_backend}
    except Exception as e:
        results = {"error": f"An unexpected error occurred: {str(e)}", "model": model_name}

    return results


def select_files_gui():
    """Open file dialogs to select selfie and ID photo using Tkinter."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    print("Please select your selfie photo...")
    selfie_path = filedialog.askopenfilename(
        title="Select Selfie Photo",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    if not selfie_path:
        print("No selfie selected.")
        return None, None

    print("Please select your ID document photo...")
    id_photo_path = filedialog.askopenfilename(
        title="Select ID Document Photo",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    if not id_photo_path:
        print("No ID document selected.")
        return selfie_path, None

    return selfie_path, id_photo_path

if __name__ == "__main__":
    # --- User Configuration for single model run ---
    # You can change CHOSEN_MODEL_NAME at the top of the script
    # or override it here for a specific run if needed.
    SELECTED_MODEL = CHOSEN_MODEL_NAME
    
    # Optional: If you have tested and found a better threshold for your CHOSEN_MODEL,
    # set it here. Otherwise, set to None to use the model's default.
    # Example: For ArcFace, default is ~0.68. If you want it stricter:
    # CUSTOM_THRESHOLD_FOR_MODEL = 0.55
    CUSTOM_THRESHOLD_FOR_MODEL = None # Use None for model's default

    print(f"INFO: Face verification will use the '{SELECTED_MODEL}' model.")
    if CUSTOM_THRESHOLD_FOR_MODEL is not None:
        print(f"INFO: Using custom threshold for decision: {CUSTOM_THRESHOLD_FOR_MODEL}")
    else:
        print(f"INFO: Using default threshold for {SELECTED_MODEL}.")
    print("-" * 30)

    selfie_path, id_photo_path = select_files_gui()

    if selfie_path and id_photo_path:
        print(f"\nVerifying Selfie: {os.path.basename(selfie_path)}")
        print(f"Against ID Photo: {os.path.basename(id_photo_path)}\n")

        print(f"--- Using Model: {SELECTED_MODEL} ---")
        verification_result = verify_faces_single_model(
            selfie_path,
            id_photo_path,
            model_name=SELECTED_MODEL, # This is now fixed based on your choice
            custom_threshold=CUSTOM_THRESHOLD_FOR_MODEL
        )
        print(json.dumps(verification_result, indent=2))

        if "error" not in verification_result:
            if verification_result["verified"]:
                print(f"\n---> RESULT: VERIFIED (Distance: {verification_result['distance']:.4f}, Threshold: {verification_result['threshold_used_for_decision']})")
            else:
                print(f"\n---> RESULT: NOT VERIFIED (Distance: {verification_result['distance']:.4f}, Threshold: {verification_result['threshold_used_for_decision']})")
        else:
            print(f"\n---> ERROR during verification: {verification_result['error']}")

    else:
        print("File selection cancelled or incomplete. Exiting.")