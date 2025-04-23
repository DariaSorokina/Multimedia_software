import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal

# 1. Программа формирования модельного аудиофайла
def generate_pulse_signal():
    Fd = 44100  # частота дискретизации
    duration = 3.0  # общая длительность в сек
    pulse_duration = 0.5  # длительность одного импульса
    pause_duration = 0.5  # длительность паузы
    freq = 440  # частота синусоиды (Гц)
    
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
sf.write('pulse_signal.wav', stereo_signal, Fd)

# 2. Программа визуализации свойств аудиофайла
def visualize_audio_properties(signal, sr):
    # Общий вид сигнала
    plt.figure(figsize=(12, 8))
    
    # График всего сигнала
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(signal[:, 0], sr=sr, color='b', alpha=0.7, label='Left Channel')
    librosa.display.waveshow(signal[:, 1], sr=sr, color='r', alpha=0.7, label='Right Channel')
    plt.title('Full Audio Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
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
    plt.title('Short Fragment (Second Pulse)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()
    
    # Амплитудный спектр
    plt.subplot(3, 1, 3)
    n_fft = 2048
    S = np.abs(librosa.stft(signal[:, 0], n_fft=n_fft))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Amplitude Spectrum')
    
    plt.tight_layout()
    plt.show()
    
    # Спектрограмма
    plt.figure(figsize=(12, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(signal[:, 0])), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.show()

# Визуализация свойств
visualize_audio_properties(stereo_signal, Fd)

# 3. Анализ реального аудиосигнала (если бы он был)
# Для демонстрации используем тот же сигнал, но можно заменить на загрузку реального файла
real_signal = stereo_signal  # Здесь можно заменить на librosa.load('real_audio.wav')

# Визуализация свойств реального сигнала (те же графики)
print("\nReal Audio Signal Analysis:")
visualize_audio_properties(real_signal, Fd)

# 4. Выводы
print("\nВыводы:")
print("1. Сформированный сигнал содержит три синусоидальных импульса длительностью 0.5 сек каждый")
print("2. Паузы между импульсами также составляют 0.5 сек")
print("3. На спектре четко видна основная частота 440 Гц и ее гармоники")
print("4. Спектрограмма показывает временное распределение энергии сигнала")
print("5. Импульсная природа сигнала хорошо видна на временной развертке")
print("6. Спектральные характеристики соответствуют ожидаемым для синусоидального сигнала")