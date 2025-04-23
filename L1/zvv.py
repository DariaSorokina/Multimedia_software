import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal

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

# Генерация и сохранение сигнала
stereo_signal, Fd = generate_pulse_signal()
sf.write('импульсный_сигнал.mp3', stereo_signal, Fd)

# 2. Программа визуализации свойств аудиофайла
def visualize_audio_properties(signal, sr):
    # Общий вид сигнала
    plt.figure(figsize=(12, 8))
    
    # График всего сигнала
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(signal[:, 0], sr=sr, color='b', alpha=0.7, label='Левый канал')
    librosa.display.waveshow(signal[:, 1], sr=sr, color='r', alpha=0.7, label='Правый канал')
    plt.title('Полный вид аудиосигнала')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.legend()
    plt.grid()
    
    # Короткий фрагмент сигнала (второй импульс)
    plt.subplot(3, 1, 2)
    start_time = 1.0
    end_time = 1.5
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    fragment = signal[start_sample:end_sample, 0]
    time_axis = np.linspace(start_time, end_time, len(fragment))
    plt.plot(time_axis, fragment, 'b')
    plt.title('Короткий фрагмент (второй импульс)')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.grid()
    
    # Амплитудный спектр
    plt.subplot(3, 1, 3)
    n_fft = 2048
    S = np.abs(librosa.stft(signal[:, 0], n_fft=n_fft))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Амплитудный спектр')
    
    plt.tight_layout()
    plt.show()
    
    # Спектрограмма
    plt.figure(figsize=(12, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(signal[:, 0])), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Спектрограмма')
    plt.tight_layout()
    plt.show()

# Визуализация свойств
visualize_audio_properties(stereo_signal, Fd)

try:
    # Загружаем реальный аудиофайл
    real_signal, sr_real = librosa.load(
        'C:/Users/dsorokina/Desktop/Rabota/pulse_signal.wav',
        sr=None,
        mono=False
    )
    
    # Если файл моно, преобразуем в стерео
    if len(real_signal.shape) == 1:
        real_signal = np.vstack((real_signal, real_signal)).T
    
    print("\nАнализ реального аудиосигнала:")
    visualize_audio_properties(real_signal, sr_real)
    
except Exception as e:
    print(f"\nОшибка при загрузке реального файла: {e}")
    print("Используется модельный сигнал для демонстрации.")
    real_signal = stereo_signal
    visualize_audio_properties(real_signal, Fd)

# 4. Выводы
print("\nРезультаты анализа:")
print("1. Сигнал содержит три синусоидальных импульса по 0.5 сек")
print("2. Паузы между импульсами составляют 0.5 сек")
print("3. Основная частота 440 Гц четко видна на спектре")
print("4. Спектрограмма показывает распределение энергии по времени")
print("5. Импульсная структура сигнала хорошо видна на графике")
print("6. Спектральные характеристики соответствуют синусоидальному сигналу")


real_signal, sr_real = librosa.load('Aut.mp3', sr=None, mono=False)
print(f"Длина сигнала: {len(real_signal)} отсчетов")
print(f"Частота дискретизации: {sr_real} Гц")
print(f"Длительность: {len(real_signal)/sr_real:.2f} сек")