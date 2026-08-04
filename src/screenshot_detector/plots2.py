import os, glob, pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from screenshot_detector import extract_features # استخراج تابع از کد خودتان

# بارگذاری مدل
with open("metadata_model.pkl", "rb") as f:
    clf = pickle.load(f)

X_test, y_true = [], []
DATASET_DIR = "./dataset"

for label, folder in [(1, "screenshots"), (0, "real_images")]:
    folder_path = os.path.join(DATASET_DIR, folder)
    for img_path in glob.glob(os.path.join(folder_path, "*.*")):
        feats = extract_features(img_path)
        if feats[0] > 0:
            X_test.append(feats)
            y_true.append(label)

X_test = np.array(X_test)
y_pred = clf.predict(X_test)
y_probs = clf.predict_proba(X_test)[:, 1]

# ۱. رسم Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Real', 'Screenshot'], 
            yticklabels=['Real', 'Screenshot'])
plt.title('Confusion Matrix - Screenshot Detector')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('screenshot_cm.png', dpi=300)
plt.show()

# ۲. رسم ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Screenshot Detector')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('screenshot_roc.png', dpi=300)
plt.show()
