import pickle
import matplotlib.pyplot as plt
import numpy as np

MODEL_PATH = "metadata_model.pkl"
FEATURE_NAMES = ["width", "height", "fnumber", "exposure", "iso", "software_flag", "has_camera_data"]

# ۱. بارگذاری مدل ذخیره شده
with open(MODEL_PATH, "rb") as f:
    clf = pickle.load(f)

# ۲. استخراج اهمیت ویژگی‌ها
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

# ۳. رسم نمودار
plt.figure(figsize=(10, 6))
plt.title("Feature Importance in Screenshot Detection (Random Forest)")
plt.bar(range(len(FEATURE_NAMES)), importances[indices], align="center", color="skyblue")
plt.xticks(range(len(FEATURE_NAMES)), [FEATURE_NAMES[i] for i in indices], rotation=45)
plt.ylabel("Importance Score")
plt.tight_layout()
plt.savefig("screenshot_feature_importance.png", dpi=300)
plt.show()
