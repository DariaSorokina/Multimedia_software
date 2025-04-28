import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk
import os

# Настройка стиля графиков
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'

# Функция для построения амплитудного спектра
def plot_amplitude_spectrum(image, title, ax=None):
    if ax is None:
        plt.figure()
        ax = plt.gca()
    
    spectrum = np.fft.fftshift(np.fft.fft2(image.astype(float)))
    spectrum_mag = 20 * np.log10(np.abs(spectrum) + 1e-9)
    
    img = ax.imshow(spectrum_mag, cmap='jet')
    ax.set_title(title)
    ax.axis('off')
    plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    
    return spectrum

# Создаем окно для выбора файла
root = Tk()
root.withdraw()

# Открываем диалог выбора файла
file_path = filedialog.askopenfilename(
    title="Выберите изображение",
    filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp"), ("All files", "*.*")]
)

if not file_path:
    print("Изображение не выбрано!")
    exit()

# Чтение изображения
input_image = cv2.imread(file_path)

# Проверка загрузки
if input_image is None:
    print(f"Ошибка загрузки изображения: {os.path.basename(file_path)}")
    exit()

# Конвертация в LAB и извлечение канала яркости
LAB_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB)
L = LAB_image[:, :, 0]  # Диапазон 0-255

# Визуализация исходного изображения и его спектра
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Исходное изображение и спектр', fontsize=14, fontweight='bold')

ax1.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
ax1.set_title('Исходное изображение (RGB)')
ax1.axis('off')

ax2.imshow(L, cmap='gray')
ax2.set_title('Канал яркости (L)')
ax2.axis('off')

# Вычисляем и отображаем амплитудный спектр
spectrum = plot_amplitude_spectrum(L, 'Амплитудный спектр (дБ)', ax3)
plt.tight_layout()
plt.show()

# Функция создания фильтра Баттерворта (оптимизированная)
def butterworth_high_pass(shape, cutoff, order=4):
    M, N = shape
    center = (M//2, N//2)
    y, x = np.ogrid[:M, :N]
    distance = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    return 1 - 1 / (1 + (distance / cutoff)**(2 * order))

# Оптимальные параметры
optimal_cutoff_ratio = 0.15  # 15% от минимального размера изображения
optimal_order = 4
d_high = round(optimal_cutoff_ratio * min(L.shape))

# Создаем и применяем фильтр
mask = butterworth_high_pass(L.shape, d_high, optimal_order)
filtered_spectr = spectrum * mask
filtered_image = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_spectr)))
filtered_image = 1 - cv2.normalize(filtered_image, None, 0, 1, cv2.NORM_MINMAX)

# Бинаризация с оптимальным порогом (метод Оцу)
_, binary_image = cv2.threshold((filtered_image*255).astype(np.uint8), 
                              0, 255, 
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Визуализация результатов фильтрации (изменено на 2x2 сетку)
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle('Результаты частотной фильтрации', fontsize=14, fontweight='bold')

# Маска фильтра
axs[0,0].imshow(mask, cmap='gray')
axs[0,0].set_title(f'Маска Баттерворта\n(срез={optimal_cutoff_ratio*100}%)')
axs[0,0].axis('off')

# Результат фильтрации
axs[0,1].imshow(filtered_image, cmap='gray')
axs[0,1].set_title('После ФВЧ Баттерворта')
axs[0,1].axis('off')

# Спектр бинаризованного изображения
plot_amplitude_spectrum(binary_image, 'Спектр после бинаризации', axs[1,0])

# Бинаризованный результат
axs[1,1].imshow(binary_image, cmap='gray')
axs[1,1].set_title('После бинаризации (Оцу)')
axs[1,1].axis('off')

plt.tight_layout()
plt.show()

# Сохранение результатов
output_dir = os.path.join(os.path.dirname(file_path), 'frequency_analysis_results')
os.makedirs(output_dir, exist_ok=True)

# Сохраняем все этапы обработки
cv2.imwrite(os.path.join(output_dir, '1_original.jpg'), input_image)
cv2.imwrite(os.path.join(output_dir, '2_L_channel.jpg'), L)
cv2.imwrite(os.path.join(output_dir, '3_filter_mask.jpg'), (mask*255).astype(np.uint8))
cv2.imwrite(os.path.join(output_dir, '4_filtered_image.jpg'), (filtered_image*255).astype(np.uint8))
cv2.imwrite(os.path.join(output_dir, '5_binary_result.jpg'), binary_image)

print(f"Результаты сохранены в папку: {output_dir}")