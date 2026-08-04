import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

VALIDATION_DIR = './data/val/'
MODEL_PATH = 'liveness_model.h5'

# ۱. بارگذاری مدل
model = load_model(MODEL_PATH)

# ۲. ساخت دیتای والیدیشن (بدون Shuffle)
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# ۳. پیش‌بینی
y_probs = model.predict(val_generator)
y_pred = (y_probs > 0.5).astype(int).ravel()
y_true = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

# نمودار ۱: Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix - Liveness Detector')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('liveness_cm.png', dpi=300)
plt.show()

# نمودار ۲: ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='green', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Liveness Detection Model')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('liveness_roc.png', dpi=300)
plt.show()

# نمودار ۳: توزیع پیش‌بینی‌ها (Confidence Distribution)
plt.figure(figsize=(8, 5))
sns.kdeplot(y_probs[y_true == 0], label='Live (Class 0)', fill=True, color='blue')
sns.kdeplot(y_probs[y_true == 1], label='Spoof (Class 1)', fill=True, color='red')
plt.axvline(0.5, color='black', linestyle='--', label='Threshold (0.5)')
plt.title('Prediction Probability Distribution')
plt.xlabel('Predicted Probability (Spoof Score)')
plt.ylabel('Density')
plt.legend()
plt.savefig('liveness_prob_dist.png', dpi=300)
plt.show()
