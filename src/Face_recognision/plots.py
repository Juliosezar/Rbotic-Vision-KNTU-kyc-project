import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from deepface import DeepFace

# نمونه فواصل استخراج شده (می‌توانید چند تصویر خود را با یک حلقه for ساده و DeepFace.verify تست کنید)
# در اینجا داده‌های تست فرضی که از خروجی ArcFace به دست می‌آید شبیه‌سازی شده:
np.random.seed(42)
same_person_distances = np.random.normal(loc=0.35, scale=0.08, size=100) # فواصل فرد یکسان (معمولا کم است)
different_person_distances = np.random.normal(loc=0.75, scale=0.1, size=100) # فواصل افراد متفاوت (معمولا زیاد است)

ARCFACE_DEFAULT_THRESHOLD = 0.68 # آستانه پیش‌فرض ArcFace برای Cosine

plt.figure(figsize=(9, 5))
sns.kdeplot(same_person_distances, fill=True, color="green", label="Same Person (Genuine Pairs)")
sns.kdeplot(different_person_distances, fill=True, color="red", label="Different Persons (Impostor Pairs)")

plt.axvline(x=ARCFACE_DEFAULT_THRESHOLD, color='black', linestyle='--', linewidth=2, 
            label=f'Threshold ({ARCFACE_DEFAULT_THRESHOLD})')

plt.title("Cosine Distance Distribution - ArcFace Model", fontsize=14)
plt.xlabel("Cosine Distance (Lower means more similar)")
plt.ylabel("Density")
plt.legend(loc="upper right")
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("face_verification_distance_dist.png", dpi=300)
plt.show()
