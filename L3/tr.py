import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from scipy import signal

plt.close('all')

# 1. Генерация модельного сигнала
def generate_test_signal(duration=3.0, fs=44100):
    """Генерация тестового сигнала с компонентами ниже и выше 3400 Гц"""
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    
    # Низкочастотная компонента (1 kHz)
    low_freq = 1000
    s_low = 0.5 * np.sin(2*np.pi*low_freq*t)
    
    # Среднечастотная компонента (2 kHz)
    mid_freq = 2000
    s_mid = 0.3 * np.sin(2*np.pi*mid_freq*t)
    
    # Высокочастотная компонента (4 kHz)
    high_freq = 4000
    s_high = 0.4 * np.sin(2*np.pi*high_freq*t)
    
    # Еще более высокая частота (6 kHz)
    very_high_freq = 6000
    s_very_high = 0.2 * np.sin(2*np.pi*very_high_freq*t)
    
    # Импульсные помехи
    impulses = np.zeros_like(t)
    for i in range(5):
        pos = int(i * duration/5 * fs)
        impulses[pos:pos+100] = 0.2 * np.random.randn(100)
    
    return s_low + s_mid + s_high + s_very_high + impulses, fs

# 2. Реализация рекурсивного ФВЧ с эллиптическим фильтром
def iir_highpass_filter(input_signal, fs, cutoff=3400, order=4):
    """Реализация ФВЧ с использованием эллиптического фильтра"""
    # Параметры эллиптического фильтра:
    rp = 1.0    # Неравномерность в полосе пропускания (дБ)
    rs = 40.0   # Ослабление в полосе подавления (дБ)
    
    # Нормализация частоты среза
    lower_frequency = cutoff
    Fd = fs
    
    # Создание эллиптического фильтра в каскадной форме (sos)
    sos = signal.ellip(order, 
                      rp, rs,
                      Wn=(lower_frequency / (Fd/2)),
                      btype='highpass', 
                      output='sos')
    
    # Применение фильтра
    output_signal = signal.sosfilt(sos, input_signal)
    
    # Расчет АЧХ фильтра
    f, h = signal.sosfreqz(sos, worN=2000, fs=fs)
    
    return output_signal, sos, f, h

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

def plot_filter_response(sos, fs):
    """Построение АЧХ фильтра"""
    f, h = signal.sosfreqz(sos, worN=2000, fs=fs)
    plt.figure(figsize=(10, 5))
    plt.semilogx(f, 20 * np.log10(np.abs(h) + 1e-9))
    plt.axvline(3400, color='k', linestyle='--', label='Граница ФВЧ (3400 Гц)')
    plt.title('АЧХ эллиптического ФВЧ (rp=1 дБ, rs=40 дБ)')
    plt.xlabel('Частота [Гц]')
    plt.ylabel('Амплитуда [дБ]')
    plt.grid(which='both', axis='both')
    plt.xlim(20, fs/2)
    plt.ylim(-80, 5)
    plt.legend()
    plt.show()

def compare_spectrograms(input_signal, output_signal, Fd, T):
    """Сравнение спектров и временных сигналов"""
    N = len(input_signal)
    t = np.linspace(0, T, N)
    
    # Вычисление спектров
    Spectr_input = np.fft.fft(input_signal)
    AS_input = np.abs(Spectr_input)
    eps = np.max(AS_input) * 1.0e-9
    S_dB_input = 20 * np.log10(AS_input + eps)
    
    Spectr_output_real = np.fft.fft(output_signal)
    S_dB_output_real = 20 * np.log10(np.abs(Spectr_output_real + eps))
    
    f = np.arange(0, Fd/2, Fd/N) # Частотная ось в Гц
    
    # Обрезка массивов по длине частотной оси
    S_dB_output_real = S_dB_output_real[:len(f)]
    S_dB_input = S_dB_input[:len(f)]
    
    # График спектров
    plt.figure(figsize=(6, 4))
    plt.semilogx(f, S_dB_input, color='b', label='Исходный спектр')
    plt.semilogx(f, S_dB_output_real, color='r', label='Фильтрованный спектр')
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(True, which='major', color='#444', linewidth=1)
    plt.grid(True, which='minor', color='#aaa', ls=':')
    
    Max_A = np.max((np.max(np.abs(Spectr_input)), np.max(np.abs(Spectr_output_real))))
    Max_dB = np.ceil(np.log10(Max_A))*20
    plt.axis([10, Fd/2, Max_dB-120, Max_dB])
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Уровень (дБ)')
    plt.title('Сравнение амплитудных спектров')
    plt.legend()
    plt.show()
    
    # Графики временных сигналов
    start_t, stop_t = 0, T
    plt.figure(figsize=(6, 3))
    plt.subplot(2, 1, 1)
    plt.plot(t, input_signal)
    plt.xlim([start_t, stop_t])
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.title('Исходный аудиосигнал')
    
    plt.subplot(2, 1, 2)
    plt.plot(t, output_signal)
    plt.xlim([start_t, stop_t])
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.title('Фильтрованный аудиосигнал')
    
    plt.tight_layout()
    plt.show()

# Основная программа
if __name__ == "__main__":
    # Анализ тестового сигнала
    test_signal, fs = generate_test_signal()
    filtered_signal, sos, f, h = iir_highpass_filter(test_signal, fs, cutoff=3400, order=4)
    
    print("Анализ тестового сигнала с эллиптическим фильтром:")
    analyze_results(test_signal, filtered_signal, fs, "(тестовый сигнал)")
    
    # Построение АЧХ фильтра
    plot_filter_response(sos, fs)
    
    # Сохранение тестовых сигналов
    sf.write('test_signal_original.wav', test_signal, fs)
    sf.write('test_signal_filtered_ellip.wav', filtered_signal, fs)
    
    # Обработка реального аудиофайла
    audio_path = 'C:/Users/dsorokina/Desktop/Rabota/L2/br.mp3'
    try:
        original_audio, fs_audio = librosa.load(audio_path, sr=None, mono=True)
        
        print("\nАнализ аудиофайла с эллиптическим фильтром:")
        filtered_audio, sos_audio, f_audio, h_audio = iir_highpass_filter(original_audio, fs_audio, cutoff=3400, order=4)
        
        analyze_results(original_audio, filtered_audio, fs_audio, "(реальный аудиофайл)")
        
        # Сравнение спектров и временных сигналов для реального аудио
        T_audio = len(original_audio)/fs_audio
        compare_spectrograms(original_audio, filtered_audio, fs_audio, T_audio)
        
        # Построение АЧХ фильтра для аудиофайла
        plot_filter_response(sos_audio, fs_audio)
        
        # Сохранение результатов
        output_path = 'C:/Users/dsorokina/Desktop/Rabota/L2/filtered_audio_ellip.wav'
        sf.write(output_path, filtered_audio, fs_audio)
        print(f"Фильтрованный аудиофайл (эллиптический фильтр) сохранен по пути: {output_path}")
        
    except Exception as e:
        print(f"Ошибка при обработке аудиофайла: {str(e)}")
        print("Продолжаем с тестовым сигналом")