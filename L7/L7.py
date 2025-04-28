import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration
from tkinter import filedialog, Tk
import os

# Функция создания маски смаза
def blur_mask(size, angle):
    k = np.zeros((size, size), dtype=np.float32)
    k[(size-1)//2, :] = np.ones(size, dtype=np.float32)
    M = cv2.getRotationMatrix2D((size/2-0.5, size/2-0.5), angle, 1.0)
    k = cv2.warpAffine(k, M, (size, size))
    return k / np.sum(k)  # Нормировка

# Функция создания маски двоения
def double_mask(size):
    mask = np.zeros((size, size), dtype=np.float32)
    mask[0, size//2] = 0.5
    mask[-1, size//2] = 0.5
    return mask

# Функция восстановления изображения
def restore_image(blurred_img, psf, noise_var=0.01, method='wiener'):
    if method == 'wiener':
        restored = np.zeros_like(blurred_img)
        for i in range(3):
            restored[:,:,i] = restoration.wiener(blurred_img[:,:,i], psf, noise_var)
    elif method == 'richardson_lucy':
        restored = np.zeros_like(blurred_img)
        for i in range(3):
            restored[:,:,i] = restoration.richardson_lucy(blurred_img[:,:,i], psf, 5)
    elif method == 'unsupervised_wiener':
        restored = np.zeros_like(blurred_img)
        for i in range(3):
            restored[:,:,i], _ = restoration.unsupervised_wiener(blurred_img[:,:,i], psf)
    return np.clip(restored, 0, 1)

# Выбор изображения
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Выберите изображение",
    filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp"), ("All files", "*.*")]
)

if not file_path:
    print("Изображение не выбрано!")
    exit()

# Загрузка изображения
input_image = cv2.imread(file_path).astype(np.float32) / 255.0
if input_image is None:
    print(f"Ошибка загрузки изображения: {os.path.basename(file_path)}")
    exit()

# ==============================================
# ЗАДАНИЕ 1: Восстановление с правильной маской
# ==============================================
print("\n=== Задание 1 ===")
psf_size = 15
angle = 30
true_psf = blur_mask(psf_size, angle)

# Создаем искаженное изображение
blurred_image = cv2.filter2D(input_image, -1, true_psf)

# Восстановление разными методами
methods = ['wiener', 'richardson_lucy', 'unsupervised_wiener']
restored_images = []

for method in methods:
    restored = restore_image(blurred_image, true_psf, method=method)
    restored_images.append(restored)

# Визуализация результатов в двух графиках
# Первый график: исходное и искаженное изображение + 1 метод восстановления
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.title('Исходное изображение')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
plt.title('Искаженное изображение')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(restored_images[0], cv2.COLOR_BGR2RGB))
plt.title(f'Восстановлено ({methods[0]})')
plt.axis('off')

plt.tight_layout()
plt.show()

# Второй график: оставшиеся 2 метода восстановления
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(restored_images[1], cv2.COLOR_BGR2RGB))
plt.title(f'Восстановлено ({methods[1]})')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(restored_images[2], cv2.COLOR_BGR2RGB))
plt.title(f'Восстановлено ({methods[2]})')
plt.axis('off')

plt.tight_layout()
plt.show()

# ==============================================
# ЗАДАНИЕ 2: Восстановление с неправильной маской
# ==============================================
print("\n=== Задание 2 ===")
# Вариант 1: Незначительное отклонение (10%)
psf_wrong1 = blur_mask(psf_size + 2, angle + 5)  # ~10% отклонение

# Вариант 2: Существенное отклонение (в 2 раза)
psf_wrong2 = blur_mask(psf_size * 2, angle * 2)

# Восстановление с неправильными масками
restored_wrong1 = restore_image(blurred_image, psf_wrong1, method='wiener')
restored_wrong2 = restore_image(blurred_image, psf_wrong2, method='wiener')

# Визуализация результатов
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
plt.title('Искаженное изображение')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(restored_wrong1, cv2.COLOR_BGR2RGB))
plt.title('Незначительное отклонение маски')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(restored_wrong2, cv2.COLOR_BGR2RGB))
plt.title('Существенное отклонение маски')
plt.axis('off')

plt.tight_layout()
plt.show()

# ==============================================
# ЗАДАНИЕ 3: Восстановление реального изображения
# ==============================================
print("\n=== Задание 3 ===")
# Создаем маску двоения (вертикальное)
double_psf = double_mask(15)

# Предполагаем, что реальное изображение уже искажено
restored_double = restore_image(input_image, double_psf, method='wiener')

# Визуализация результатов
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.title('Исходное изображение')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(restored_double, cv2.COLOR_BGR2RGB))
plt.title('Восстановленное (двоение)')
plt.axis('off')

plt.tight_layout()
plt.show()

# Сохранение результатов
output_dir = 'restoration_results'
os.makedirs(output_dir, exist_ok=True)

cv2.imwrite(f'{output_dir}/0_original.jpg', input_image*255)
cv2.imwrite(f'{output_dir}/1_blurred.jpg', blurred_image*255)
for i, method in enumerate(methods):
    cv2.imwrite(f'{output_dir}/1_restored_{method}.jpg', restored_images[i]*255)
cv2.imwrite(f'{output_dir}/2_restored_wrong1.jpg', restored_wrong1*255)
cv2.imwrite(f'{output_dir}/2_restored_wrong2.jpg', restored_wrong2*255)
cv2.imwrite(f'{output_dir}/3_restored_double.jpg', restored_double*255)

print(f"\nВсе результаты сохранены в папку: {output_dir}")