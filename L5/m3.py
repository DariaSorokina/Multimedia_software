import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, exposure
import os
from tkinter import filedialog
from tkinter import Tk

def show_image_with_hist(image, title, img_pos, hist_pos, rows=2, cols=2):
    """Отображение изображения с соответствующей гистограммой"""
    # Изображение
    plt.subplot(rows, cols, img_pos)
    if len(image.shape) == 2:  # Grayscale
        plt.imshow(image, cmap='gray')
    else:  # Color
        plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    
    # Гистограмма
    plt.subplot(rows, cols, hist_pos)
    if len(image.shape) == 2:  # Grayscale
        hist, bin_edges = np.histogram(image.flatten(), bins=256, range=(0, 1))
        plt.plot(bin_edges[:-1], hist, color='gray')
    else:  # Color
        for i, color_channel in enumerate(('red', 'green', 'blue')):
            channel = image[:, :, i].flatten()
            hist, bin_edges = np.histogram(channel, bins=256, range=(0, 1))
            plt.plot(bin_edges[:-1], hist, color=color_channel)
    plt.title(f'Гистограмма {title}')
    plt.grid(True, alpha=0.3)

# Выбор изображения
root = Tk()
root.withdraw()
image_path = filedialog.askopenfilename(title="Выберите изображение")

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Файл не найден: {image_path}")

try:
    image = io.imread(image_path)
    # Удаляем альфа-канал, если он есть (4-й канал)
    if image.shape[-1] == 4:
        image = image[..., :3]  # Берем только RGB каналы
    # Нормализуем изображение к диапазону [0, 1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255
    print(f"Изображение загружено. Размер: {image.shape}")
except Exception as e:
    print(f"Ошибка загрузки изображения: {e}")
    exit()

# Конвертируем в grayscale для equalize_hist
gray_image = color.rgb2gray(image) if len(image.shape) == 3 else image

# Создаем фигуру для отображения результатов (2 строки × 2 столбца)
plt.figure(figsize=(15, 10))

# 1. Исходное изображение (позиции 1 и 2)
show_image_with_hist(gray_image, 'Исходное изображение', 1, 2)

# 2. Equalize Hist с n=7 (nbins=128)
equalized = exposure.equalize_hist(gray_image, nbins=2**7)
show_image_with_hist(equalized, 'Equalize Hist (nbins=128)', 3, 4)

plt.tight_layout()
plt.show()

# Сохранение результатов
output_dir = os.path.join(os.path.dirname(image_path), 'selected_results')
os.makedirs(output_dir, exist_ok=True)

# Сохраняем исходное изображение
orig_path = os.path.join(output_dir, 'original.jpg')
io.imsave(orig_path, (255 * gray_image).clip(0, 255).astype(np.uint8), quality=95)

# Сохраняем обработанное изображение
equalized_path = os.path.join(output_dir, 'equalized_n7.jpg')
io.imsave(equalized_path, (255 * equalized).clip(0, 255).astype(np.uint8), quality=95)

print(f"Сохранено: {orig_path}")
print(f"Сохранено: {equalized_path}")