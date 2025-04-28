import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk
import os

# Проверяем доступные стили и выбираем подходящий
available_styles = plt.style.available
preferred_styles = ['seaborn-v0_8', 'seaborn', 'ggplot', 'classic']
selected_style = None

for style in preferred_styles:
    if style in available_styles:
        selected_style = style
        break

if selected_style:
    plt.style.use(selected_style)
else:
    plt.style.use('default')  # Используем стиль по умолчанию, если предпочтительные недоступны

# Настройка параметров графиков
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'

# Функция для отображения спектра
def plot_spectrum(image, title, ax):
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    spectrum_mag = 20 * np.log10(np.abs(spectrum) + 1e-9)
    im = ax.imshow(spectrum_mag, cmap='jet')
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

# Загрузка изображения
image = cv2.imread(file_path)
if image is None:
    print(f"Ошибка загрузки изображения: {os.path.basename(file_path)}")
    exit()

# Конвертация в grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Методы выделения границ
## 1. Кэнни
canny_low = cv2.Canny(image, 50, 150)
canny_high = cv2.Canny(image, 100, 200)

## 2. Собеля
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
sobel = np.sqrt(sobel_x**2 + sobel_y**2)

## 3. Лапласиан
laplacian = cv2.Laplacian(gray, cv2.CV_64F)

# Нормализация
def normalize(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

canny_low = normalize(canny_low)
canny_high = normalize(canny_high)
sobel = normalize(sobel)
laplacian = normalize(laplacian)

# Создаем фигуру для сравнения методов
fig, axs = plt.subplots(4, 2, figsize=(16, 24))
fig.suptitle('Анализ методов выделения границ', y=1.02, fontsize=18, fontweight='bold')

# Исходное изображение
axs[0,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0,0].set_title('Исходное изображение', pad=15)
axs[0,0].axis('off')

plot_spectrum(gray, 'Спектр исходного изображения', axs[0,1])

# Кэнни (низкие пороги)
axs[1,0].imshow(canny_low, cmap='gray')
axs[1,0].set_title('Детектор Кэнни (пороги 50/150)', pad=15)
axs[1,0].axis('off')

plot_spectrum(canny_low, 'Спектр: Кэнни (50/150)', axs[1,1])

# Кэнни (высокие пороги)
axs[2,0].imshow(canny_high, cmap='gray')
axs[2,0].set_title('Детектор Кэнни (пороги 100/200)', pad=15)
axs[2,0].axis('off')

plot_spectrum(canny_high, 'Спектр: Кэнни (100/200)', axs[2,1])

# Собеля
axs[3,0].imshow(sobel, cmap='gray')
axs[3,0].set_title('Оператор Собеля (ядро 5x5)', pad=15)
axs[3,0].axis('off')

plot_spectrum(sobel, 'Спектр: Собеля', axs[3,1])

plt.tight_layout()

# Отдельная фигура для Лапласиана
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig2.suptitle('Оператор Лапласа', y=1.02, fontsize=18, fontweight='bold')

ax1.imshow(laplacian, cmap='gray')
ax1.set_title('Результат обработки', pad=15)
ax1.axis('off')

plot_spectrum(laplacian, 'Спектр: Лапласиан', ax2)

plt.tight_layout()
plt.show()

# Сохранение результатов
output_dir = os.path.join(os.path.dirname(file_path), 'edge_detection_results')
os.makedirs(output_dir, exist_ok=True)

# Сохраняем основные результаты
fig.savefig(os.path.join(output_dir, 'comparison.png'), dpi=300, bbox_inches='tight')
fig2.savefig(os.path.join(output_dir, 'laplacian.png'), dpi=300, bbox_inches='tight')

# Сохраняем отдельные изображения
results = {
    'original': image,
    'canny_low': cv2.cvtColor(canny_low, cv2.COLOR_GRAY2BGR),
    'canny_high': cv2.cvtColor(canny_high, cv2.COLOR_GRAY2BGR),
    'sobel': cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR),
    'laplacian': cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
}

for name, img in results.items():
    output_path = os.path.join(output_dir, f'{name}.png')
    cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"Сохранено: {output_path}")

print(f"\nВсе результаты сохранены в папку: {output_dir}")
print(f"Использованный стиль графиков: {selected_style or 'default'}")