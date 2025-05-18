import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
import os

def create_color_mask(hsv_img, lower, upper, morph_iters=2):
    """
    Создает маску цвета с морфологической обработкой в HSV пространстве
    :param hsv_img: Изображение в HSV
    :param lower: Нижняя граница HSV
    :param upper: Верхняя граница HSV
    :param morph_iters: Количество итераций морфологической обработки
    :return: Бинарная маска
    """
    mask = cv2.inRange(hsv_img, lower, upper)
    
    # Морфологическая обработка для устранения шума
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iters)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iters)
    
    return mask

def get_dominant_hsv_values(hsv_img, mask):
    """
    Получает доминирующие значения HSV в области маски
    :param hsv_img: Изображение в HSV
    :param mask: Бинарная маска
    :return: Средние значения H, S, V
    """
    if np.count_nonzero(mask) == 0:
        return 0, 0, 0
    
    masked = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)
    h, s, v = cv2.split(masked)
    
    # Вычисляем средние значения только для ненулевых пикселей
    h_mean = np.mean(h[h > 0])
    s_mean = np.mean(s[s > 0])
    v_mean = np.mean(v[v > 0])
    
    return h_mean, s_mean, v_mean

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

    # Загрузка изображения
    image = cv2.imread(file_path)
    if image is None:
        print("Ошибка загрузки изображения!")
        return

    # Конвертация в HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Диапазоны цветов в HSV (можно настроить под конкретное изображение)
    # Желтый цвет
    yellow_lower = np.array([30, 100, 100])
    yellow_upper = np.array([60, 255, 255])
    
    # Пурпурный цвет
    magenta_lower = np.array([130, 100, 100])
    magenta_upper = np.array([165, 255, 255])

    # Создание масок с морфологической обработкой
    yellow_mask = create_color_mask(image_hsv, yellow_lower, yellow_upper)
    magenta_mask = create_color_mask(image_hsv, magenta_lower, magenta_upper)

    # Копия для результата
    result = image_hsv.copy()

    # Если найдены оба цвета, выполняем замену
    if np.any(yellow_mask) and np.any(magenta_mask):
        # Получаем средние значения S и V для каждого цвета
        y_h, y_s, y_v = get_dominant_hsv_values(image_hsv, yellow_mask)
        m_h, m_s, m_v = get_dominant_hsv_values(image_hsv, magenta_mask)
        
        # Желтый -> Пурпурный
        result[yellow_mask > 0, 0] = 150 # Hue для пурпурного
        result[yellow_mask > 0, 1] = min(m_s * 1.2, 255)  # Увеличиваем насыщенность
        result[yellow_mask > 0, 2] = min(m_v * 1.1, 255)  # Увеличиваем яркость
        
        # Пурпурный -> Желтый
        result[magenta_mask > 0, 0] = 30  # Hue для желтого
        result[magenta_mask > 0, 1] = min(y_s * 1.2, 255)  # Увеличиваем насыщенность
        result[magenta_mask > 0, 2] = min(y_v * 1.1, 255)  # Увеличиваем яркость

    # Конвертируем обратно в BGR для сохранения и RGB для отображения
    result_bgr = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_HSV2BGR)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Визуализация
    plt.figure(figsize=(15, 5))
    
    # Исходное изображение
    plt.subplot(1, 3, 1)
    plt.imshow(original_rgb)
    plt.title('Исходное изображение')
    plt.axis('off')
    
    # Результат
    plt.subplot(1, 3, 2)
    plt.imshow(result_rgb)
    plt.title('После замены цветов')
    plt.axis('off')
    
    # Маски (желтый и пурпурный)
    plt.subplot(1, 3, 3)
    mask_display = np.zeros_like(original_rgb)
    mask_display[yellow_mask > 0] = [255, 255, 0]  # Желтый в RGB
    mask_display[magenta_mask > 0] = [255, 0, 255]  # Пурпурный в RGB
    plt.imshow(mask_display)
    plt.title('Обнаруженные цвета')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    # Сохранение результата
    output_dir = os.path.dirname(file_path)
    output_name = f"swapped_{os.path.basename(file_path)}"
    output_path = os.path.join(output_dir, output_name)
    
    cv2.imwrite(output_path, result_bgr)
    print(f"Результат сохранен: {output_path}")

if __name__ == "__main__":
    main()