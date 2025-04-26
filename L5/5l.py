import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, exposure # type: ignore
from skimage.color import rgb2lab, lab2rgb # type: ignore
import os
from tkinter import filedialog
from tkinter import Tk

# Укажите путь к изображению
root = Tk()
root.withdraw()
image_path = filedialog.askopenfilename(title="Выберите изображение")

# Проверка существования файла
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Файл не найден: {image_path}")

# 1. Загрузка и отображение исходного изображения
try:
    image = io.imread(image_path)
    print(f"Изображение загружено. Размер: {image.shape}")
except Exception as e:
    print(f"Ошибка загрузки изображения: {e}")
    exit()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Исходное изображение')
plt.axis('off')

# 2. Конвертация в LAB и обработка яркости
LAB_image = rgb2lab(image)
L = LAB_image[:, :, 0] / 100  # Нормализация яркостного канала

# 3. Коррекция динамического диапазона
L_out = exposure.rescale_intensity(
    L,
    in_range=(np.min(L), np.max(L)),
    out_range=(0, 1)
).astype(np.float32)

# 4. Гамма-коррекция (попробуйте разные значения gamma)
gamma_value = 1.2  # Можно изменять (0.5 - осветление, 2.0 - затемнение)
L_out_gam = exposure.adjust_gamma(L_out, gamma=gamma_value)

# 5. Обратная конвертация в RGB
LAB_image[:, :, 0] = L_out_gam * 100
image_out = lab2rgb(LAB_image)

# 6. Отображение обработанного изображения
plt.subplot(1, 2, 2)
plt.imshow(image_out)
plt.title(f'Обработанное (γ={gamma_value})')
plt.axis('off')
plt.tight_layout()
plt.show()

# 7. Построение гистограммы яркости
histogram, bin_edges = np.histogram(L_out, bins=256, range=(0, 1))

plt.figure(figsize=(8, 4))
plt.bar(bin_edges[:-1], histogram, width=0.003, color='gray')
plt.title('Гистограмма распределения яркости')
plt.xlabel('Яркость (нормированная)')
plt.ylabel('Количество пикселей')
plt.grid(True, alpha=0.3)
plt.show()

# 8. Сохранение результата
output_path = os.path.join(os.path.dirname(image_path), 'processed_image.jpg')
try:
    # Конвертация в 8-битный формат перед сохранением
    image_out_8bit = (255 * image_out).clip(0, 255).astype(np.uint8)
    io.imsave(output_path, image_out_8bit, quality=95)
    print(f"Изображение успешно сохранено: {output_path}")
except Exception as e:
    print(f"Ошибка сохранения: {e}")

# Дополнительная визуализация каналов LAB
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(LAB_image[:, :, 0], cmap='gray')
plt.title('L-канал (Яркость)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(LAB_image[:, :, 1], cmap='coolwarm')
plt.title('A-канал')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(LAB_image[:, :, 2], cmap='coolwarm')
plt.title('B-канал')
plt.axis('off')
plt.tight_layout()
plt.show()