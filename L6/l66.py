import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk
import os

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-white')  # Белый фон с темными линиями
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'

def invert_edges(edges):
    #Инвертирует изображение границ (делает темные границы на белом фоне)
    return 255 - edges

def normalize(img):
    #Нормализует значения массива к диапазону от 0 до 255.
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max != img_min:
        normalized_img = ((img - img_min) / (img_max - img_min)) * 255
    else:
        normalized_img = np.zeros_like(img)
    return normalized_img.astype(np.uint8)

def process_laplacian(image, threshold=30):
    #Обработка Лапласиана с пороговой бинаризацией
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    _, binary = cv2.threshold(laplacian, threshold, 255, cv2.THRESH_BINARY)
    return invert_edges(binary)

def plot_spectrum(image, title, ax):
    #Отображение спектра с темными линиями на светлом фоне
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    spectrum_mag = 20 * np.log10(np.abs(spectrum) + 1e-9)
    im = ax.imshow(spectrum_mag, cmap='viridis')  # Измененная цветовая карта
    ax.set_title(title, pad=15)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

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

# Загрузка и конвертация
image = cv2.imread(file_path)
if image is None:
    print(f"Ошибка загрузки изображения: {os.path.basename(file_path)}")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Обработка границ с инверсией
canny_low = invert_edges(cv2.Canny(gray, 50, 150))
canny_high = invert_edges(cv2.Canny(gray, 100, 200))

sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
sobel = invert_edges(normalize(np.sqrt(sobel_x**2 + sobel_y**2)))

laplacian = process_laplacian(gray)  # Специальная обработка для Лапласиана

# Создание фигур
fig, axs = plt.subplots(4, 2, figsize=(16, 24))
fig.suptitle('Сравнение методов выделения границ', y=1.02, fontsize=18, fontweight='bold')

# Исходное изображение
axs[0,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0,0].set_title('Исходное изображение', pad=15)
axs[0,0].axis('off')

plot_spectrum(gray, 'Спектр исходного изображения', axs[0,1])

# Кэнни
axs[1,0].imshow(canny_low, cmap='gray', vmin=0, vmax=255)
axs[1,0].set_title('Кэнни (пороги 50/150)', pad=15)
axs[1,0].axis('off')

plot_spectrum(canny_low, 'Спектр: Кэнни (50/150)', axs[1,1])

axs[2,0].imshow(canny_high, cmap='gray', vmin=0, vmax=255)
axs[2,0].set_title('Кэнни (пороги 100/200)', pad=15)
axs[2,0].axis('off')

plot_spectrum(canny_high, 'Спектр: Кэнни (100/200)', axs[2,1])

# Собеля
axs[3,0].imshow(sobel, cmap='gray', vmin=0, vmax=255)
axs[3,0].set_title('Собеля (ядро 5x5)', pad=15)
axs[3,0].axis('off')

plot_spectrum(sobel, 'Спектр: Собеля', axs[3,1])

plt.tight_layout()

# Фигура для Лапласиана
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig2.suptitle('Оператор Лапласа с пороговой обработкой', y=1.02, fontsize=18, fontweight='bold')

ax1.imshow(laplacian, cmap='gray', vmin=0, vmax=255)
ax1.set_title('Лапласиан', pad=15)
ax1.axis('off')

plot_spectrum(laplacian, 'Спектр: Лапласиан', ax2)

plt.tight_layout()
plt.show()

# Сохранение результатов
output_dir = os.path.join(os.path.dirname(file_path), 'edge_detection_results2')
os.makedirs(output_dir, exist_ok=True)

results = {
    'original': image,
    'canny_low': canny_low,
    'canny_high': canny_high,
    'sobel': sobel,
    'laplacian': laplacian
}

for name, img in results.items():
    output_path = os.path.join(output_dir, f'{name}.png')
    cv2.imwrite(output_path, img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Сохранено: {output_path}")

print(f"\nВсе результаты сохранены в: {output_dir}")