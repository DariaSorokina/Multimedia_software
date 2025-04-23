import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# Параметры сигнала
SAMPLE_RATE = 44100  # Частота дискретизации (Гц)
DURATION = 3.0       # Общая длительность (3 импульса по 0.5 сек + 2 паузы по 0.5 сек)
PULSE_DUR = 0.5      # Длительность одного импульса (сек)
PAUSE_DUR = 0.5      # Длительность паузы (сек)
FREQUENCY = 440      # Частота синусоиды (Гц)

def generate_signal():
    """Генерация сигнала с тремя синусоидальными импульсами"""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    signal = np.zeros_like(t)
    
    # Генерация трех импульсов
    for i in range(3):
        start = i * (PULSE_DUR + PAUSE_DUR)
        end = start + PULSE_DUR
        mask = (t >= start) & (t < end)
        signal[mask] = np.sin(2 * np.pi * FREQUENCY * t[mask])
    
    return signal

def create_stereo(signal):
    """Создание стерео сигнала (одинаковый в обоих каналах)"""
    return np.column_stack((signal, signal))

def normalize(signal):
    """Нормировка сигнала"""
    max_val = np.max(np.abs(signal))
    return signal if max_val == 0 else signal / max_val

def plot_signal(signal, title):
    """Визуализация сигнала"""
    t = np.linspace(0, DURATION, len(signal), endpoint=False)
    plt.figure(figsize=(12, 4))
    plt.plot(t, signal[:, 0], label='Левый канал')
    plt.plot(t, signal[:, 1], '--', alpha=0.7, label='Правый канал')
    plt.title(title)
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.legend()
    plt.grid()
    plt.show()

# Генерация и обработка сигнала
mono_signal = generate_signal()
stereo_signal = create_stereo(mono_signal)
normalized_signal = normalize(stereo_signal)

# Визуализация
plot_signal(normalized_signal, 'Три синусоидальных импульса с паузами')

# Сохранение в файл
OUTPUT_FILE = 'sinus_pulses.wav'
sf.write(OUTPUT_FILE, normalized_signal, SAMPLE_RATE)
print(f"Аудиофайл сохранён как: {OUTPUT_FILE}")