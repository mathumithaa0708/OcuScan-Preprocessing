import os
import cv2
import numpy as np

input_path = r"C:\Users\niven\OneDrive\Desktop\Ocuscan\data\DRIVE\Images\Traning"

import os

print("Checking path:", input_path)
print("Path exists:", os.path.exists(input_path))

green_path = "output/green"
clahe_path = "output/clahe"
denoise_path = "output/denoised"
fused_path = "output/fused"
compare_path = "output/comparison"

os.makedirs(green_path, exist_ok=True)
os.makedirs(clahe_path, exist_ok=True)
os.makedirs(denoise_path, exist_ok=True)
os.makedirs(fused_path, exist_ok=True)
os.makedirs(compare_path, exist_ok=True)
image_list = os.listdir(input_path)
print("All folders created successfully")

for img_name in image_list:
    img_path = os.path.join(input_path, img_name)
    image = cv2.imread(img_path)
    image = cv2.resize(image, (512, 512))
    green = image[:, :, 1]
    cv2.imwrite(os.path.join(green_path, img_name), green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(green, cv2.MORPH_TOPHAT, kernel)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(green)
    cv2.imwrite(os.path.join(clahe_path, img_name), clahe_img)
    denoised = cv2.GaussianBlur(clahe_img, (5, 5), 0)
    cv2.imwrite(os.path.join(denoise_path, img_name), denoised)
    fused = cv2.addWeighted(clahe_img, 0.7, tophat, 0.3, 0)
    cv2.imwrite(os.path.join(fused_path, img_name), fused)
    original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    comparison = np.hstack((original_gray, fused))
    cv2.imwrite(os.path.join(compare_path, img_name), comparison)
print("preprocessing completed successfully.")


