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

LAB_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB)
# это мы перевели в LAB
L = LAB_image[:, :, 0]/100 # выделяем матрицу яркости
# Визуализация исходного изображения
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.title('Исходное изображение')
plt.axis('off')
plt.tight_layout()
plt.show()
# Двумерное преобразование Фурье и амплитудный спектр
Spectr = np.fft.fftshift(np.fft.fft2(L))
eps = np.max(np.abs(Spectr)) * 1e-9
Spectr_dB = 20 * np.log10(np.abs(Spectr) + eps)
plt.imshow(Spectr_dB, cmap='jet')
plt.title('Амплитудный спектр (дБ)')
plt.axis('off')
plt.tight_layout()
plt.show()
