
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk
import os

# Создаем окно для выбора файла
root = Tk()
root.withdraw()  # Скрываем основное окно

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
L = LAB_image[:, :, 0]  # Диапазон 0-255 (не нужно делить на 100)

# Визуализация исходного изображения
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.title('Исходное изображение (RGB)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(L, cmap='gray')
plt.title('Канал яркости (L)')
plt.axis('off')
plt.tight_layout()
plt.show()

# Двумерное преобразование Фурье
Spectr = np.fft.fftshift(np.fft.fft2(L.astype(float)))
eps = np.max(np.abs(Spectr)) * 1e-9

# Функция создания фильтра Баттерворта
def butterworth_high_pass(shape, cutoff, order=4):
    M, N = shape
    mask = np.zeros((M, N))
    center_m, center_n = M // 2, N // 2
    
    for m in range(M):
        for n in range(N):
            d = np.sqrt((m - center_m)**2 + (n - center_n)**2)
            mask[m, n] = 1 - 1 / (1 + (d / cutoff)**(2 * order))
    
    return mask

# Параметры для сравнения (три разных частоты среза)
cutoff_ratios = [0.1, 0.2, 0.3]  # 10%, 20% и 30% от минимального размера
order = 4  # Порядок фильтра

plt.figure(figsize=(15, 10))

for i, ratio in enumerate(cutoff_ratios, 1):
    # Вычисляем частоту среза
    d_high = round(ratio * min(L.shape))
    
    # Создаем маску
    mask = butterworth_high_pass(L.shape, d_high, order)
    
    # Применяем фильтр
    filtered_spectr = Spectr * mask
    filtered_image = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_spectr)))
    
    # Нормализация и инверсия
    filtered_image = cv2.normalize(filtered_image, None, 0, 1, cv2.NORM_MINMAX)
    filtered_image = 1 - filtered_image  # Инверсия для лучшей визуализации
    
    # Визуализация результатов
    # Маска фильтра
    plt.subplot(3, 3, (i-1)*3 + 1)
    plt.imshow(mask, cmap='gray')
    plt.title(f'Маска Баттерворта\n(срез={ratio*100}%, порядок={order})')
    plt.axis('off')
    
    # Спектр после фильтрации
    plt.subplot(3, 3, (i-1)*3 + 2)
    spectr_dB = 20 * np.log10(np.abs(filtered_spectr) + eps)
    plt.imshow(spectr_dB, cmap='jet')
    plt.title(f'Спектр после фильтра\n(срез={ratio*100}%)')
    plt.axis('off')
    
    # Результат фильтрации
    plt.subplot(3, 3, (i-1)*3 + 3)
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f'Результат фильтрации\n(срез={ratio*100}%)')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Сохранение результатов
output_dir = 'butterworth_results'
import os
os.makedirs(output_dir, exist_ok=True)

for i, ratio in enumerate(cutoff_ratios):
    d_high = round(ratio * min(L.shape))
    mask = butterworth_high_pass(L.shape, d_high, order)
    filtered_spectr = Spectr * mask
    filtered_image = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_spectr)))
    filtered_image = 255 * (1 - cv2.normalize(filtered_image, None, 0, 1, cv2.NORM_MINMAX))
    
    cv2.imwrite(f'{output_dir}/butterworth_{int(ratio*100)}percent.jpg', filtered_image.astype('uint8'))

print(f"Результаты сохранены в папку: {output_dir}")