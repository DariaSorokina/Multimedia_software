import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import os

def load_and_analyze_audio(file_path):
    """
    Загрузка и анализ аудиофайла с визуализацией характеристик
    """
    try:
        # Загрузка аудиофайла
        signal, sr = librosa.load(file_path, sr=None, mono=False)
        
        # Проверка и преобразование в стерео при необходимости
        if signal.ndim == 1:
            signal = np.vstack([signal, signal])
        signal = signal.T  # Преобразуем в форму (samples, channels)

        # Визуализация характеристик
        visualize_audio_characteristics(signal, sr, os.path.basename(file_path))
        
        return True
        
    except Exception as e:
        print(f"Ошибка при анализе файла {file_path}: {str(e)}")
        return False

def visualize_audio_characteristics(signal, sr, title=""):
    """
    Визуализация характеристик аудиосигнала
    """
    plt.figure(figsize=(14, 10))
    
    # 1. Полный вид сигнала
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(signal[:, 0], sr=sr, color='b', label='Левый канал')
    if signal.shape[1] > 1:
        librosa.display.waveshow(signal[:, 1], sr=sr, color='r', alpha=0.7, label='Правый канал')
    plt.title(f'Полный вид сигнала: {title}')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.legend()
    plt.grid()
    
    # 2. Детальный фрагмент (10 мс)
    plt.subplot(3, 1, 2)
    fragment_start = int(sr * 1.0)  # Начинаем с 1 секунды
    fragment_length = int(sr * 0.01)  # 10 мс фрагмент
    fragment = signal[fragment_start:fragment_start+fragment_length, 0]
    time_axis = np.linspace(1.0, 1.01, len(fragment), endpoint=False)
    plt.plot(time_axis, fragment, 'b')
    plt.title('Детальный вид фрагмента (10 мс)')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
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
        plt.title(f'Короткий фрагмент')
        plt.xlabel('Время (с)')
        plt.ylabel('Амплитуда')
        plt.grid()
    else:
        plt.text(0.5, 0.5, 'Слишком короткий сигнал для фрагмента', 
                ha='center', va='center')
    
    # 3. Амплитудный спектр
    plt.subplot(3, 1, 3)
    n_fft = min(2048, len(signal)//2)
    S = np.abs(librosa.stft(signal[:, 0], n_fft=n_fft))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Амплитудный спектр (линейная шкала)')
    
    plt.tight_layout()
    plt.show()
    
    # 4. Спектрограмма (отдельный график)
    plt.figure(figsize=(14, 5))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(signal[:, 0])), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Спектрограмма (логарифмическая шкала частот)')
    plt.tight_layout()
    plt.show()

def generate_test_signal():
    """
    Генерация тестового сигнала с тремя импульсами
    """
    Fd = 44100  # Частота дискретизации
    duration = 3.0  # Общая длительность
    pulse_duration = 0.5  # Длительность импульса
    pause_duration = 0.5  # Длительность паузы
    freq = 440  # Частота синусоиды
    
    t = np.arange(0, duration, 1/Fd)
    signal = np.zeros(len(t))
    
    # Генерация трех импульсов
    for i in range(3):
        start = i * (pulse_duration + pause_duration)
        end = start + pulse_duration
        mask = (t >= start) & (t < end)
        signal[mask] = np.sin(2 * np.pi * freq * t[mask])
    
    # Создание стерео сигнала
    stereo_signal = np.vstack([signal, signal]).T
    
    # Нормировка
    stereo_signal = stereo_signal / np.max(np.abs(stereo_signal))
    
    return stereo_signal, Fd

def main():
    # 1. Генерация тестового сигнала
    test_signal, sr = generate_test_signal()
    output_file = 'test_signal.wav'
    sf.write(output_file, test_signal, sr)
    print(f"Тестовый сигнал сохранен как: {output_file}")
    
    # 2. Анализ сгенерированного файла
    print("\nАнализ тестового сигнала:")
    if not load_and_analyze_audio(output_file):
        print("Не удалось проанализировать тестовый сигнал")
    
    # 3. Анализ реального файла
    user_file = 'C:/Users/dsorokina/Desktop/Rabota/Steve.mp3'
    if os.path.exists(user_file):
        print(f"\nАнализ музыкального файла: {user_file}")
        load_and_analyze_audio(user_file)
    else:
        print(f"\nФайл не найден: {user_file}")

if __name__ == "__main__":
    main()