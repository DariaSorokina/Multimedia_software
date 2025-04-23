import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import os

def load_and_analyze_audio(file_path):
    """
    Загрузка и анализ аудиофайла с комплексной визуализацией
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
    Комплексная визуализация характеристик аудиосигнала
    """
    N = len(signal)
    
    # 1. Создаем фигуру с тремя субплoтами
    plt.figure(figsize=(14, 12))
    plt.suptitle(f'Анализ аудиосигнала: {title}', y=1.02, fontsize=14)
    
    # 1.1. Временной график (полный сигнал)
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(signal[:, 0], sr=sr, color='b', label='Левый канал')
    librosa.display.waveshow(signal[:, 1], sr=sr, color='r', alpha=0.7, label='Правый канал')
    plt.title('Временная область сигнала', pad=20)
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 1.2. Детальный фрагмент (второй импульс)
    plt.subplot(3, 1, 2)
    start_time = 1.0  # Начало второго импульса
    end_time = 1.5    # Конец второго импульса
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    fragment = signal[start_sample:end_sample, 0]
    time_axis = np.linspace(start_time, end_time, len(fragment))
    
    plt.plot(time_axis, fragment, 'b', linewidth=1.5)
    plt.title('Детальный вид второго импульса (0.5 сек)', pad=20)
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 1.3. Амплитудный спектр (логарифмическая шкала)
    plt.subplot(3, 1, 3)
    Spectr = np.fft.fft(signal[:, 0])
    AS = np.abs(Spectr)
    eps = np.max(AS) * 1.0e-9
    S_dB = 20 * np.log10(AS + eps)
    
    f = np.arange(0, sr/2, sr/N)
    S_dB = S_dB[:len(f)]
    
    plt.semilogx(f, S_dB, 'b', linewidth=1)
    plt.grid(True, which='both')
    plt.minorticks_on()
    plt.grid(True, which='major', color='#444', linewidth=0.8)
    plt.grid(True, which='minor', color='#aaa', linestyle=':')
    
    Max_dB = np.ceil(np.max(S_dB)/20)*20
    plt.axis([20, sr/2, Max_dB-80, Max_dB+5])  # Оптимизированные границы
    
    plt.title('Амплитудный спектр (логарифмическая шкала)', pad=20)
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Уровень (дБ)')
    
    plt.tight_layout()
    plt.show()
    
    # 2. Спектрограмма (отдельный график)
    plt.figure(figsize=(14, 6))
    n_fft = min(2048, len(signal)//4)
    hop_length = n_fft // 4
    
    D = librosa.amplitude_to_db(np.abs(librosa.stft(
        signal[:, 0], n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    
    librosa.display.specshow(D, sr=sr, hop_length=hop_length,
                           x_axis='time', y_axis='log',
                           cmap='viridis')
    
    plt.colorbar(format='%+2.0f dB', pad=0.02)
    plt.title('Спектрограмма (логарифмическая шкала частот)', pad=20)
    plt.xlabel('Время (с)')
    plt.ylabel('Частота (Гц)')
    plt.tight_layout()
    plt.show()

def generate_test_signal():
    """
    Генерация тестового сигнала с тремя синусоидальными импульсами
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
    # 1. Анализ тестового сигнала
    test_signal, sr = generate_test_signal()
    test_file = 'pulse_signal.wav'
    sf.write(test_file, test_signal, sr)
    
    print(f"Анализ тестового сигнала ({test_file}):")
    load_and_analyze_audio(test_file)
    
    # 2. Анализ реального файла
    user_file = r'C:\Users\dsorokina\Desktop\Rabota\Steve.mp3'
    if os.path.exists(user_file):
        print(f"\nАнализ файла: {user_file}")
        load_and_analyze_audio(user_file)
    else:
        print(f"\nФайл не найден: {user_file}")
        print("Используется тестовый сигнал для демонстрации")

if __name__ == "__main__":
    main()