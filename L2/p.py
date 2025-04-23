import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from scipy import signal

plt.close('all')

# 1. Генерация модельного сигнала для демонстрации фильтра
def generate_test_signal(duration=3.0, fs=44100):
    """Генерация тестового сигнала с компонентами ниже и выше 3400 Гц"""
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    
    # Низкочастотная компонента (1 kHz)
    low_freq = 1000
    s_low = 0.5 * np.sin(2*np.pi*low_freq*t)
    
    # Высокочастотная компонента (4 kHz)
    high_freq = 4000
    s_high = 0.3 * np.sin(2*np.pi*high_freq*t)
    
    # Импульсные помехи
    impulses = np.zeros_like(t)
    for i in range(5):
        pos = int(i * duration/5 * fs)
        impulses[pos:pos+100] = 0.2 * np.random.randn(100)
    
    return s_low + s_high + impulses, fs

# 2. Реализация ФВЧ через Фурье-фильтрацию
def fourier_highpass_filter(input_signal, fs, cutoff=3400):
    """Реализация ФВЧ через прямое/обратное БПФ"""
    N = len(input_signal)
    
    # Вычисление БПФ
    spectr = np.fft.fft(input_signal)
    freqs = np.fft.fftfreq(N, 1/fs)
    
    # Создание передаточной функции ФВЧ
    W = np.ones(N, dtype=float)
    W[np.abs(freqs) < cutoff] = 0
    
    # Применение фильтра
    filtered_spectr = spectr * W
    
    # Обратное БПФ
    output_signal = np.real(np.fft.ifft(filtered_spectr))
    
    return output_signal, spectr, filtered_spectr, freqs

# 3. Анализ и визуализация результатов
def analyze_results(original, filtered, fs, title_suffix=""):
    """Визуализация временных, спектральных характеристик и спектрограмм"""
    
    # Временные графики
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(original))/fs, original)
    plt.title(f'Исходный сигнал {title_suffix}')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.grid()
    
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(filtered))/fs, filtered)
    plt.title(f'Фильтрованный сигнал {title_suffix}')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    # Спектры
    plt.figure(figsize=(12, 6))
    f = np.linspace(0, fs/2, len(original)//2)
    
    # Исходный спектр
    S_orig = np.abs(np.fft.fft(original)[:len(f)])
    S_orig_db = 20*np.log10(S_orig + 1e-9)
    
    # Фильтрованный спектр
    S_filt = np.abs(np.fft.fft(filtered)[:len(f)])
    S_filt_db = 20*np.log10(S_filt + 1e-9)
    
    plt.semilogx(f, S_orig_db, 'b', label='Исходный')
    plt.semilogx(f, S_filt_db, 'r', label='После ФВЧ')
    plt.axvline(3400, color='k', linestyle='--', label='Граница ФВЧ (3400 Гц)')
    plt.title('Сравнение амплитудных спектров')
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Уровень (дБ)')
    plt.legend()
    plt.grid(True, which='both')
    plt.xlim(20, fs/2)
    plt.ylim(-60, max(S_orig_db)+10)
    plt.show()
    
    # Спектрограммы
    plt.figure(figsize=(12, 10))
    
    n_fft = 2048
    hop_length = n_fft // 4
    
    plt.subplot(2, 1, 1)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(original, n_fft=n_fft, 
                                                  hop_length=hop_length)), 
                              ref=np.max)
    librosa.display.specshow(D, sr=fs, hop_length=hop_length,
                           x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Спектрограмма исходного сигнала {title_suffix}')
    plt.ylim(20, fs/2)
    
    plt.subplot(2, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(filtered, n_fft=n_fft,
                                                  hop_length=hop_length)), 
                              ref=np.max)
    librosa.display.specshow(D, sr=fs, hop_length=hop_length,
                           x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Спектрограмма после ФВЧ {title_suffix}')
    plt.ylim(20, fs/2)
    
    plt.tight_layout()
    plt.show()

# Основная программа
if __name__ == "__main__":
    # Анализ тестового сигнала
    test_signal, fs = generate_test_signal()
    filtered_signal, orig_spectr, filt_spectr, freqs = fourier_highpass_filter(test_signal, fs, 3400)
    
    print("Анализ тестового сигнала:")
    analyze_results(test_signal, filtered_signal, fs, "(тестовый сигнал)")
    
    # Сохранение тестовых сигналов
    sf.write('test_signal_original.wav', test_signal, fs)
    sf.write('test_signal_filtered.wav', filtered_signal, fs)
    
    # Обработка реального аудиофайла
    audio_path = 'C:/Users/dsorokina/Desktop/Rabota/L2/br.mp3'
    try:
        original_audio, fs_audio = librosa.load(audio_path, sr=None, mono=True)
        
        print("\nАнализ  аудиофайла барабан:")
        filtered_audio, _, _, _ = fourier_highpass_filter(original_audio, fs_audio, 3400)
        
        analyze_results(original_audio, filtered_audio, fs_audio, "(аудиофайл барабан)")
        
        # Сохранение результатов
        sf.write(r'C:/Users/dsorokina/Desktop/Rabota/L2/filtered_audio.mp3', filtered_audio, fs_audio)
        
    except Exception as e:
        print(f"Ошибка при обработке аудиофайла: {str(e)}")
        print("Продолжаем с тестовым сигналом")
