import os
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

TRAIN_DIR = './data/train/'  # Expecting two subfolders: "Spoof" and "Live"
VALIDATION_DIR = './data/val/'
MODEL_PATH = 'liveness_model.h5'

def train_model():
    """Train the liveness detection model using existing dataset structure"""
    # Data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.2,
        horizontal_flip=True,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    # Model architecture 
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='Adam',
        metrics=['accuracy']
    )

    # Train model
    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        verbose=1
    )

    # Save model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

class LivenessDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Liveness Detector")
        self.model = load_model(MODEL_PATH)
        
        self.create_widgets()

    def create_widgets(self):
        self.btn_load = tk.Button(
            self.root, 
            text="Load Image", 
            command=self.load_image,
            padx=20, pady=10
        )
        self.btn_load.pack(pady=20)

        self.lbl_image = tk.Label(self.root)
        self.lbl_image.pack()

        self.lbl_result = tk.Label(self.root, text="", font=('Helvetica', 14))
        self.lbl_result.pack(pady=20)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                img = Image.open(file_path)
                img = img.resize((300, 300))
                photo = ImageTk.PhotoImage(img)
                self.lbl_image.config(image=photo)
                self.lbl_image.image = photo
                
                # Preprocess and predict
                prediction = self.predict_liveness(file_path)
                result = "Live" if prediction < 0.5 else "Spoof"
                confidence = (1 - prediction) if result == "Live" else prediction
                self.lbl_result.config(
                    text=f"Result: {result} (Confidence: {confidence:.2%})",
                    fg="green" if result == "Live" else "red"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Invalid image file: {e}")

    def predict_liveness(self, image_path):
        """Predict if image is live or spoof"""
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(150, 150))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  
        img_array /= 255.0  

        prediction = self.model.predict(img_array)
        return prediction[0][0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model')
    args = parser.parse_args()

    if args.train:
        print("Starting model training...")
        train_model()
    else:
        root = tk.Tk()
        app = LivenessDetectorApp(root)
        root.mainloop()
