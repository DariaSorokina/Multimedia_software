import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import random

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

# 1. Звук огня (треск пламени)
def generate_fire_sound():
    # Основной шум огня
    b, a = create_filter('bandpass', [200, 2000])
    fire_sound = apply_filter(white_noise, b, a)
    
    # Добавляем "трески" - случайные импульсы
    for _ in range(200):
        pos = random.randint(0, num_samples-100)
        fire_sound[pos:pos+100] += 0.5 * np.random.randn(100)
    
    # Модуляция для эффекта мерцания
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * np.arange(num_samples) / sampling_rate)
    return fire_sound * modulation

# 2. Звук дождя
def generate_rain_sound():
    # Основной шум дождя
    b, a = create_filter('bandpass', [500, 5000])
    rain_sound = apply_filter(white_noise, b, a)
    
    # Добавляем капли - случайные всплески
    for _ in range(500):
        pos = random.randint(0, num_samples-50)
        rain_sound[pos:pos+50] += 0.3 * np.random.randn(50)
    
    # Медленная модуляция интенсивности
    modulation = 0.7 + 0.3 * np.sin(2 * np.pi * 0.1 * np.arange(num_samples) / sampling_rate)
    return rain_sound * modulation

# 3. Пение птиц (чириканье)
def generate_birds_sound():
    bird_sound = np.zeros(num_samples)
    
    # Генерируем несколько птичьих трелей
    for _ in range(20):
        start = random.randint(0, num_samples - 10000)
        duration_samples = random.randint(1000, 5000)
        freq = random.randint(2000, 6000)
        
        # Создаем чирикающий звук
        t = np.arange(duration_samples) / sampling_rate
        chirp = np.sin(2 * np.pi * freq * t) * np.hanning(duration_samples)
        
        # Модулируем частоту для эффекта трели
        freq_mod = freq * (1 + 0.2 * np.sin(2 * np.pi * 5 * t))
        chirp = np.sin(2 * np.pi * freq_mod * t) * np.hanning(duration_samples)
        
        bird_sound[start:start+duration_samples] += chirp * 0.5
    
    # Добавляем фоновый шум листвы
    b, a = create_filter('highpass', [2000])
    background = apply_filter(white_noise, b, a) * 0.1
    
    return bird_sound + background

# Нормализация сигналов
def normalize(signal):
    return np.int16((signal / np.max(np.abs(signal))) * 32767)

# Генерация и сохранение звуков
fire_sound = generate_fire_sound()
rain_sound = generate_rain_sound()
birds_sound = generate_birds_sound()

wavfile.write('C:/Users/dsorokina/Desktop/Rabota/L4/fire_sound.wav', sampling_rate, normalize(fire_sound))
wavfile.write('C:/Users/dsorokina/Desktop/Rabota/L4/rain_sound.wav', sampling_rate, normalize(rain_sound))
wavfile.write('C:/Users/dsorokina/Desktop/Rabota/L4/birds_sound.wav', sampling_rate, normalize(birds_sound))

print("Аудиофайлы успешно созданы: fire_sound.wav, rain_sound.wav, birds_sound.wav")