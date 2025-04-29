import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
import os

def create_color_mask(hsv_img, lower, upper):
    """Создает маску цвета с морфологической обработкой"""
    mask = cv2.inRange(hsv_img, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask

def main():
    # Настройка выбора файла
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Выберите изображение",
        filetypes=[("Изображения", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    
    if not file_path:
        print("Изображение не выбрано!")
        return

    # Загрузка и проверка изображения
    image = cv2.imread(file_path)
    if image is None:
        print("Ошибка загрузки изображения!")
        return

    # Конвертация в HSV и RGB
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Диапазоны цветов в HSV
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    magenta_lower = np.array([145, 100, 100])
    magenta_upper = np.array([165, 255, 255])

    # Создание масок
    yellow_mask = create_color_mask(image_hsv, yellow_lower, yellow_upper)
    magenta_mask = create_color_mask(image_hsv, magenta_lower, magenta_upper)

    # Копия для результата
    result = image_hsv.copy()

    # Меняем цвета местами
    if np.any(yellow_mask) and np.any(magenta_mask):
        # Получаем средние значения S и V для каждого цвета
        yellow_s = np.mean(image_hsv[yellow_mask > 0, 1])
        yellow_v = np.mean(image_hsv[yellow_mask > 0, 2])
        magenta_s = np.mean(image_hsv[magenta_mask > 0, 1])
        magenta_v = np.mean(image_hsv[magenta_mask > 0, 2])
        
        # Желтый -> Пурпурный (H=150)
        result[yellow_mask > 0, 0] = 150  # Hue
        result[yellow_mask > 0, 1] = magenta_s * 1.0  # Saturation
        result[yellow_mask > 0, 2] = magenta_v * 0.9  # Value
        
        # Пурпурный -> Желтый (H=30)
        result[magenta_mask > 0, 0] = 30  # Hue
        result[magenta_mask > 0, 1] = yellow_s * 1.0  # Saturation
        result[magenta_mask > 0, 2] = yellow_v * 1.1  # Value

    # Конвертируем обратно в RGB
    result_rgb = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # Визуализация
    plt.figure(figsize=(15, 5))
    titles = ['Исходное', 'После замены', 'Маски цветов']
    images = [
        image_rgb,
        result_rgb,
        np.dstack([yellow_mask, np.zeros_like(yellow_mask), magenta_mask])
    ]
    
    for i, (title, img) in enumerate(zip(titles, images), 1):
        plt.subplot(1, 3, i)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    # Сохранение результата
    output_dir = os.path.dirname(file_path)
    output_name = f"swapped_{os.path.basename(file_path)}"
    output_path = os.path.join(output_dir, output_name)
    
    cv2.imwrite(output_path, cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR))
    print(f"Результат сохранен: {output_path}")

if __name__ == "__main__":
    main()