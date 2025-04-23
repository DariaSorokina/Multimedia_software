import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import os

# Путь к аудиофайлу
AUDIO_PATH = r'C:\Users\dsorokina\Desktop\Rabota\pulse_signal.wav'

def generate_pulse_signal():
    """Генерация тестового сигнала с тремя импульсами"""
    Fd = 44100
    duration = 3.0
    pulse_duration = 0.5
    pause_duration = 0.5
    freq = 440
    
    t = np.arange(0, duration, 1/Fd)
    pulse_signal = np.zeros(len(t))
    
    for i in range(3):
        start = i * (pulse_duration + pause_duration)
        end = start + pulse_duration
        mask = (t >= start) & (t < end)
        pulse_signal[mask] = np.sin(2 * np.pi * freq * t[mask])
    
    stereo_signal = np.column_stack((pulse_signal, pulse_signal))
    stereo_signal = stereo_signal / np.max(np.abs(stereo_signal))
    
    return stereo_signal, Fd

def analyze_audio(file_path):
    """Анализ и визуализация аудиофайла"""
    try:
        # Загрузка файла
        signal, sr = sf.read(file_path)
        if signal.ndim == 1:
            signal = np.column_stack((signal, signal))
            
        N = len(signal)
        
        # 1. Временная область
        plt.figure(figsize=(14, 10))
        
        # 1.1 Полный сигнал
        plt.subplot(3, 1, 1)
        librosa.display.waveshow(signal[:, 0], sr=sr, color='b', label='Левый канал')
        librosa.display.waveshow(signal[:, 1], sr=sr, color='r', alpha=0.7, label='Правый канал')
        plt.title('Временная область сигнала')
        plt.xlabel('Время (с)')
        plt.ylabel('Амплитуда')
        plt.legend()
        plt.grid()
        
        # 1.2 Детальный фрагмент
        plt.subplot(3, 1, 2)
        start_sample = int(1.0 * sr)  # Начало второго импульса
        end_sample = int(1.5 * sr)    # Конец второго импульса
        fragment = signal[start_sample:end_sample, 0]
        time_axis = np.linspace(1.0, 1.5, len(fragment))
        plt.plot(time_axis, fragment, 'b')
        plt.title('Детальный вид (второй импульс)')
        plt.xlabel('Время (с)')
        plt.ylabel('Амплитуда')
        plt.grid()
        
        # 2. Амплитудный спектр (левый канал)
        plt.subplot(3, 1, 3)
        Spectr = np.fft.fft(signal[:, 0])
        AS = np.abs(Spectr)
        eps = np.max(AS) * 1.0e-9
        S_dB = 20 * np.log10(AS + eps)
        
        f = np.arange(0, sr/2, sr/N)
        S_dB = S_dB[:len(f)]
        
        plt.semilogx(f, S_dB)
        plt.grid(True)
        plt.minorticks_on()
        plt.grid(True, which='major', color='#444', linewidth=1)
        plt.grid(True, which='minor', color='#aaa', ls=':')
        
        Max_dB = np.ceil(np.max(S_dB)/20)*20
        plt.axis([10, sr/2, Max_dB-100, Max_dB])
        plt.xlabel('Частота (Гц)')
        plt.ylabel('Уровень (дБ)')
        plt.title('Амплитудный спектр')
        
        plt.tight_layout()
        plt.show()
        
        # 3. Спектрограмма
        plt.figure(figsize=(14, 5))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(signal[:, 0])), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Спектрограмма')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")

if __name__ == "__main__":
    # Создаем тестовый сигнал (если нужно)
    # signal, sr = generate_pulse_signal()
    # sf.write('pulse_signal.wav', signal, sr)
    
    # Анализируем существующий файл
    if os.path.exists(AUDIO_PATH):
        analyze_audio(AUDIO_PATH)
    else:
        print(f"Файл не найден: {AUDIO_PATH}")
        print("Создаем тестовый сигнал...")
        signal, sr = generate_pulse_signal()
        sf.write('pulse_signal.wav', signal, sr)
        analyze_audio('pulse_signal.wav')