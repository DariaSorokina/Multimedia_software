import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, exposure
import os
from tkinter import filedialog
from tkinter import Tk

def show_image_with_hist(image, title, img_pos, hist_pos, rows=5, cols=2):
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

# Создаем фигуру для отображения результатов (5 строк × 2 столбца)
plt.figure(figsize=(15, 20))

# 1. Исходное изображение (позиции 1 и 2)
show_image_with_hist(image, 'Исходное изображение', 1, 2)

# 2. Rescale Intensity (позиции 3 и 4)
rescaled = exposure.rescale_intensity(image, out_range=(0, 9))
show_image_with_hist(rescaled, 'Rescale Intensity', 3, 4)

# 3. Gamma Correction (γ=1.9) (позиции 5 и 6)
gamma_corrected = exposure.adjust_gamma(image, gamma=1.9)
show_image_with_hist(gamma_corrected, 'Gamma Correction (γ=1.9)', 5, 6)

# 4. Equalize Hist (глобальное выравнивание) (позиции 7 и 8)
if len(image.shape) == 3:
    # Конвертируем в grayscale, если изображение цветное
    gray_image = color.rgb2gray(image)
    equalized = exposure.equalize_hist(gray_image)
else:
    equalized = exposure.equalize_hist(image)
show_image_with_hist(equalized, 'Equalize Hist', 7, 8)

# 5. Adaptive Equalize Hist (адаптивное выравнивание) (позиции 9 и 10)
adaptive = exposure.equalize_adapthist(image, clip_limit=0.05)
show_image_with_hist(adaptive, 'Adaptive Equalize Hist', 9, 10)

plt.tight_layout()
plt.show()

# Сохранение результатов
output_dir = os.path.join(os.path.dirname(image_path), 'processed_results')
os.makedirs(output_dir, exist_ok=True)

methods = {
    'original': image,
    'rescaled': rescaled,
    'gamma_corrected': gamma_corrected,
    'equalized': equalized,
    'adaptive': adaptive
}

for name, img in methods.items():
    try:
        output_path = os.path.join(output_dir, f'{name}.jpg')
        if img.dtype != np.uint8:
            img = (255 * img).clip(0, 255).astype(np.uint8)
        io.imsave(output_path, img, quality=95)
        print(f"Сохранено: {output_path}")
    except Exception as e:
        print(f"Ошибка сохранения {name}: {e}")