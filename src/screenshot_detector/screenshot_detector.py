import os
import glob
import argparse
import pickle
import numpy as np
from PIL import Image, ExifTags, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Configuration ---
DATASET_DIR = "./dataset"  # Expecting two subfolders: "screenshots" and "real_images"
MODEL_PATH = "metadata_model.pkl"  # Save model as pickle file
FEATURE_NAMES = ["width", "height", "fnumber", "exposure", "iso", "software_flag", "has_camera_data"]

# --- Utility Functions ---
def get_exif_data(image_path):
    """Extracts EXIF data using PIL and returns a dictionary with tag names as keys."""
    try:
        image = Image.open(image_path)
        exif_raw = image._getexif()
        exif = {}
        if exif_raw is not None:
            for tag, value in exif_raw.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                exif[decoded] = value
        return exif, image.size  # image.size returns (width, height)
    except Exception as e:
        print(f"Error reading EXIF from {image_path}: {e}")
        return {}, (0, 0)

def safe_float(value):
    """Converts a value to float safely (handles tuple fraction values)."""
    try:
        # Some EXIF values are stored as (numerator, denominator)
        if isinstance(value, tuple) and len(value) == 2 and value[1] != 0:
            return float(value[0]) / float(value[1])
        return float(value)
    except Exception:
        return 0.0

def extract_features(image_path):
    """
    Extract metadata-based features from the image.
    Features include:
      - width and height (resolution)
      - FNumber (aperture)
      - ExposureTime (as a float)
      - ISOSpeedRatings
      - software_flag: 1 if the 'Software' tag contains "screenshot", else 0
      - has_camera_data: 1 if typical camera EXIF fields (FNumber, ExposureTime, ISO) are present and >0, else 0
    """
    exif, (width, height) = get_exif_data(image_path)
    fnumber = safe_float(exif.get("FNumber", 0))
    exposure = safe_float(exif.get("ExposureTime", 0))
    iso = safe_float(exif.get("ISOSpeedRatings", 0))
    
    # Check the Software tag for the substring "screenshot" (case-insensitive)
    software = str(exif.get("Software", "")).lower()
    software_flag = 1 if "screenshot" in software else 0
    
    # Determine if camera-specific data is present
    has_camera_data = 1 if (fnumber > 0 and exposure > 0 and iso > 0) else 0
    
    features = np.array([width, height, fnumber, exposure, iso, software_flag, has_camera_data], dtype=float)
    return features

# --- Training Function ---
def train_model():
    X = []
    y = []
    # Assume directory structure: dataset/screenshots and dataset/real_images
    for label, folder in [(1, "screenshots"), (0, "real_images")]:
        folder_path = os.path.join(DATASET_DIR, folder)
        image_files = glob.glob(os.path.join(folder_path, "*.*"))
        for img_path in image_files:
            features = extract_features(img_path)
            # Only add if resolution is nonzero (valid image)
            if features[0] > 0 and features[1] > 0:
                X.append(features)
                y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data into training and testing (for evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a RandomForest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation Accuracy: {acc*100:.2f}%")
    
    # Save the trained model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved to {MODEL_PATH}")

# --- GUI Function ---
def launch_gui():
    # Load trained model
    try:
        with open(MODEL_PATH, "rb") as f:
            clf = pickle.load(f)
    except Exception as e:
        print("Error loading model. Please train the model first using: python screenshot_detector.py --train")
        return

    def load_image():
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                features = extract_features(file_path).reshape(1, -1)
                prediction = clf.predict(features)[0]
                # For confidence, we use predict_proba if available
                proba = clf.predict_proba(features)[0]
                conf = proba[1] if prediction == 1 else proba[0]
                result = "Screenshot" if prediction == 1 else "Real Image"
                result_label.config(text=f"Result: {result} (Confidence: {conf:.2%})")
                
                # Display the image in the GUI
                pil_img = Image.open(file_path)
                pil_img = pil_img.resize((300, 300))
                photo = ImageTk.PhotoImage(pil_img)
                img_label.config(image=photo)
                img_label.image = photo
            except Exception as ex:
                messagebox.showerror("Error", str(ex))
    
    root = tk.Tk()
    root.title("Screenshot Detector (Metadata Only)")
    btn_load = tk.Button(root, text="Load Image", command=load_image, padx=20, pady=10)
    btn_load.pack(pady=20)
    img_label = tk.Label(root)
    img_label.pack()
    result_label = tk.Label(root, text="", font=("Helvetica", 14))
    result_label.pack(pady=20)
    root.mainloop()

# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model using metadata features")
    args = parser.parse_args()
    
    if args.train:
        train_model()
    else:
        launch_gui()
