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

# Преобразование в RGB
image_rgb = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB)

# Создание маски красного цвета в RGB пространстве
# Красный цвет имеет высокие значения в R-канале и низкие в G и B
red_mask = (image_rgb[:,:,0] > 0.9) & (image_rgb[:,:,1] < 0.9) & (image_rgb[:,:,2] < 0.9)

# Создаем grayscale версию изображения
gray_value = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
gray_rgb = cv2.cvtColor(gray_value, cv2.COLOR_GRAY2RGB)

# Заменяем красные области на grayscale
image_rgb_new = image_rgb.copy()
image_rgb_new[red_mask] = gray_rgb[red_mask]

# Визуализация
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Исходное изображение')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image_rgb_new)
plt.title('Без красного (серый)')
plt.axis('off')
plt.show()

# Сохранение результата
output_dir = os.path.dirname(file_path)
output_filename = os.path.splitext(os.path.basename(file_path))[0] + '_no_red.jpg'
output_path = os.path.join(output_dir, output_filename)

image_bgr_new = cv2.cvtColor(image_rgb_new, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_path, (image_bgr_new * 255).astype(np.uint8))

print(f"Изображение сохранено: {output_path}")