import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
import os

# 1. Программа формирования модельного аудиофайла
def generate_pulse_signal():
    Fd = 44100  # Частота дискретизации
    duration = 3.0  # Общая длительность в сек
    pulse_duration = 0.5  # Длительность одного импульса
    pause_duration = 0.5  # Длительность паузы
    freq = 440  # Частота синусоиды (Гц)
    
    # Создаем временную ось
    t = np.arange(0, duration, 1/Fd)
    
    # Создаем сигнал с тремя импульсами
    pulse_signal = np.zeros(len(t))
    for i in range(3):
        start = i * (pulse_duration + pause_duration)
        end = start + pulse_duration
        mask = (t >= start) & (t < end)
        pulse_signal[mask] = np.sin(2 * np.pi * freq * t[mask])
    
    # Создаем стерео сигнал (одинаковый в обоих каналах)
    stereo_signal = np.vstack((pulse_signal, pulse_signal)).T
    
    # Нормировка
    Norm = np.max(np.abs(stereo_signal))
    if Norm != 0:
        stereo_signal = stereo_signal / Norm
    
    return stereo_signal, Fd

# 2. Улучшенная программа визуализации свойств аудиофайла
def visualize_audio_properties(signal, sr, title_suffix=""):
    # Проверка и подготовка сигнала
    if signal.ndim == 1:
        signal = np.vstack((signal, signal)).T
    
    # Общий вид сигнала
    plt.figure(figsize=(12, 10))
    
    # График всего сигнала
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(signal[:, 0], sr=sr, color='b', alpha=0.7, label='Левый канал')
    if signal.shape[1] > 1:  # Если есть правый канал
        librosa.display.waveshow(signal[:, 1], sr=sr, color='r', alpha=0.7, label='Правый канал')
    plt.title(f'Полный вид аудиосигнала {title_suffix}')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.legend()
    plt.grid()
    
    # Короткий фрагмент сигнала (середина файла)
    plt.subplot(3, 1, 2)
    duration = len(signal)/sr
    start_time = max(0, duration/2 - 0.025)  # 50 мс вокруг середины
    end_time = min(duration, duration/2 + 0.025)
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    if end_sample > start_sample:  # Проверка на корректность диапазона
        fragment = signal[start_sample:end_sample, 0]
        time_axis = np.linspace(start_time, end_time, len(fragment))
        plt.plot(time_axis, fragment, 'b')
        plt.title(f'Короткий фрагмент {title_suffix}')
        plt.xlabel('Время (с)')
        plt.ylabel('Амплитуда')
        plt.grid()
    else:
        plt.text(0.5, 0.5, 'Слишком короткий сигнал для фрагмента', 
                ha='center', va='center')
    
    # Амплитудный спектр и спектрограмма
    plt.subplot(3, 1, 3)
    n_fft = min(2048, len(signal)//2)  # Автоподбор размера окна
    hop_length = n_fft//4
    
    if n_fft > 16:  # Минимальный разумный размер окна
        S = np.abs(librosa.stft(signal[:, 0], n_fft=n_fft, hop_length=hop_length))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, 
                               x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Амплитудный спектр {title_suffix}')
    else:
        plt.text(0.5, 0.5, 'Слишком короткий сигнал для спектрального анализа', 
                ha='center', va='center')
    
    plt.tight_layout()
    plt.show()
    
    # Отдельная спектрограмма с большим разрешением
    if n_fft > 16:
        plt.figure(figsize=(12, 6))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(signal[:, 0], n_fft=n_fft, 
                                                    hop_length=hop_length)), 
                                  ref=np.max)
        librosa.display.specshow(D, sr=sr, hop_length=hop_length,
                               x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Спектрограмма {title_suffix}')
        plt.tight_layout()
        plt.show()

# Генерация и сохранение тестового сигнала
stereo_signal, Fd = generate_pulse_signal()
sf.write('импульсный_сигнал.wav', stereo_signal, Fd)

# Визуализация тестового сигнала
print("Анализ тестового сигнала:")
visualize_audio_properties(stereo_signal, Fd, "(тестовый)")

# Анализ реального аудиофайла
audio_path = 'C:/Users/dsorokina/Desktop/Rabota/pulse_signal.wav'
try:
    # Проверка существования файла
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Файл не найден: {audio_path}")
    
    # Загрузка с обработкой возможных ошибок
    real_signal, sr_real = librosa.load(audio_path, sr=None, mono=False)
    
    # Проверка загруженных данных
    if len(real_signal) == 0:
        raise ValueError("Файл пуст или не содержит данных")
    
    print(f"\nАнализ реального аудиосигнала ({audio_path}):")
    print(f"Длительность: {len(real_signal)/sr_real:.2f} сек")
    print(f"Частота дискретизации: {sr_real} Гц")
    print(f"Форма сигнала: {real_signal.shape}")
    
    # Визуализация
    visualize_audio_properties(real_signal, sr_real, "(Autechre - Gantz Graf)")
    
except Exception as e:
    print(f"\nОшибка при обработке реального файла: {e}")
    print("Используется тестовый сигнал для демонстрации.")
    visualize_audio_properties(stereo_signal, Fd, "(резервный)")

# Выводы
print("\nРезультаты анализа:")
print("1. Тестовый сигнал содержит три синусоидальных импульса по 0.5 сек")
print("2. Реальный аудиофайл анализируется с адаптивными параметрами")
print("3. Все графики автоматически подстраиваются под длину сигнала")
print("4. Код устойчив к ошибкам загрузки файлов")