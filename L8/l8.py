import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk
import os

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
image_in = cv2.imread(file_path).astype(np.float32) / 255.0
if image_in is None:
    print(f"Ошибка загрузки изображения: {os.path.basename(file_path)}")
    exit()

# Преобразование в RGB и HSV
image_rgb = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB)
image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

# Создание маски зеленого цвета
lower_green = np.array([35, 0, 0])
upper_green = np.array([90, 255, 255])
mask_green = cv2.inRange(image_hsv, lower_green, upper_green)

# Замена зеленого на серый (S=0)
image_hsv_new = image_hsv.copy()
image_hsv_new[mask_green > 0, 1] = 0
image_rgb_new = cv2.cvtColor(image_hsv_new, cv2.COLOR_HSV2RGB)

# Визуализация
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Исходное изображение')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image_rgb_new)
plt.title('Без зеленого (серый)')
plt.axis('off')
plt.show()

# Сохранение (исправленный путь)
output_path = r"C:\Users\dsorokina\Desktop\Rabota\Multimedia_software\Multimedia_software\L8\image_out.jpg"
image_bgr_new = cv2.cvtColor(image_rgb_new, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_path, image_bgr_new * 255)  # Умножаем на 255 для корректного диапазона [0, 255]

print(f"Изображение сохранено: {output_path}")