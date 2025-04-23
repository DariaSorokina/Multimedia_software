import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter


# Основные параметры сигнала
sampling_rate = 44100   # Частота дискретизации (Гц)
duration = 10           # Длительность сигнала (секунды)
num_samples = sampling_rate * duration  # Общее количество образцов

# Генерация белого шума
white_noise = np.random.randn(num_samples)  # Генерация белого шума (нормальное распределение)


# Функция для создания фильтра Баттерворта
def create_filter(filter_type, cutoff_freqs, order=2):
    nyquist = 0.5 * sampling_rate  # Частота Найквиста
    normalized_cutoff = [freq / nyquist for freq in cutoff_freqs]  # Нормализация частот
    b, a = butter(order, normalized_cutoff, btype=filter_type)  # Создание фильтра Баттерворта
    return b, a


# Функция для применения фильтра к сигналу
def apply_filter(signal, b, a):
    return lfilter(b, a, signal)  # Применение фильтра к сигналу


# Аудиофайл 1: Звук горелки (без изменений)
b, a = create_filter('low', [300])  # Граничная частота 300 Гц
burner_sound = apply_filter(white_noise + white_noise * 0.1, b, a)  # Добавление белого шума

# Аудиофайл 2: Шум прибоя
b, a = create_filter('bandpass', [700, 1300])  # Полоса пропускания 700–1300 Гц
ocean_sound = apply_filter(white_noise, b, a)
modulation_frequency = 0.2  # Частота модуляции (например, 0.2 Гц)
modulation = (np.sin(2 * np.pi * modulation_frequency * np.arange(num_samples) / sampling_rate) + 1) / 2
ocean_sound = ocean_sound * modulation  # Применение модуляции к шуму

# Аудиофайл 3: Спуск колеса
b, a = create_filter('high', [5000])  # Граничная частота 5000 Гц
tire_sound = apply_filter(white_noise, b, a)


# Нормализация сигналов для предотвращения искажений при сохранении
def normalize(signal):
    return np.int16((signal / np.max(np.abs(signal))) * 32767)  # Приведение к диапазону int16

# Сохранение аудиофайлов
wavfile.write('C:/Users/dsorokina/Desktop/Rabota/L4/burner_sound.wav', sampling_rate, normalize(burner_sound))  # Звук горелки
wavfile.write('C:/Users/dsorokina/Desktop/Rabota/L4/ocean_sound.wav', sampling_rate, normalize(ocean_sound))    # Шум прибоя
wavfile.write('C:/Users/dsorokina/Desktop/Rabota/L4/tire_sound.wav', sampling_rate, normalize(tire_sound))      # Спуск колеса