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
# Конвертация в черно-белое
L = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Визуализация исходного изображения
plt.figure(figsize=(8, 4))
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.title('Исходное изображение')
plt.axis('off')
plt.tight_layout()
plt.show()

# Двумерное центрированное БПФ
Spectr = np.fft.fftshift(np.fft.fft2(L))
eps = np.max(np.abs(Spectr)) * 1e-9

# Функция для применения ФВЧ с заданной частотой среза
def apply_high_pass_filter(image, freq_ratio):
    M, N = image.shape
    d_high = round(freq_ratio * min(M, N))
    
    # Создаем маску
    mask = np.ones((M, N))
    
    # Определяем границы для четных/нечетных размеров
    if M % 2 == 0: 
        m_range = (M//2-d_high, M//2+d_high)
    else: 
        m_range = (M//2-d_high, M//2+d_high+1)
        
    if N % 2 == 0: 
        n_range = (N//2-d_high, N//2+d_high)
    else: 
        n_range = (N//2-d_high, N//2+d_high+1)
    
    mask[m_range[0]:m_range[1], n_range[0]:n_range[1]] = 0
    
    # Применяем фильтр (ИСПРАВЛЕННАЯ СТРОКА)
    filtered_spectr = Spectr * mask
    filtered_image = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_spectr)).real)  # Закрыты все скобки
    filtered_image = 1 - cv2.normalize(filtered_image, None, 0, 1, cv2.NORM_MINMAX)
    
    return mask, filtered_spectr, filtered_image

# Применяем фильтр с двумя разными частотами среза
freq_ratios = [0.15, 0.30]  # 15% и 30% от минимального размера изображения

plt.figure(figsize=(15, 10))

for i, ratio in enumerate(freq_ratios, 1):
    mask, filtered_spectr, filtered_image = apply_high_pass_filter(L, ratio)
    
    # Визуализация маски
    plt.subplot(2, 3, (i-1)*3 + 1)
    plt.imshow(mask, cmap='gray')
    plt.title(f'Маска ФВЧ (срез {ratio*100}%)')
    plt.axis('off')
    
    # Визуализация спектра
    plt.subplot(2, 3, (i-1)*3 + 2)
    spectr_dB = 20*np.log10(np.abs(filtered_spectr)+eps)
    spectr_dB[0,0] = 20*np.log10(np.max(np.abs(Spectr)))  # Нормализация цветовой шкалы
    plt.imshow(spectr_dB, cmap='jet')
    plt.title(f'Спектр после ФВЧ ({ratio*100}%)')
    plt.axis('off')
    
    # Визуализация результата
    plt.subplot(2, 3, (i-1)*3 + 3)
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f'Результат фильтра ({ratio*100}%)')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Сохранение результатов
for i, ratio in enumerate(freq_ratios):
    _, _, filtered_image = apply_high_pass_filter(L, ratio)
    cv2.imwrite(f'high_pass_{int(ratio*100)}percent.jpg', 
               (filtered_image * 255).astype('uint8'))