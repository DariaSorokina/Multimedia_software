import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import os

def analyze_audio_file(file_path):
    """
    Анализ и визуализация свойств аудиофайла
    """
    try:
        # 1. Загрузка аудиофайла
        audio_data, sample_rate = sf.read(file_path)
        
        # Проверка и преобразование в стерео при необходимости
        if audio_data.ndim == 1:
            audio_data = np.column_stack((audio_data, audio_data))
        
        # 2. Настройка параметров визуализации
        plt.figure(figsize=(12, 8))
        plt.suptitle('Анализ аудиофайла: ' + os.path.basename(file_path), y=1.02)
        
        # 3. Временной график сигнала
        plt.subplot(2, 1, 1)
        time = np.arange(len(audio_data)) / sample_rate
        plt.plot(time, audio_data[:, 0], 'b', label='Левый канал', alpha=0.7)
        plt.plot(time, audio_data[:, 1], 'r', label='Правый канал', alpha=0.5)
        plt.title('Временная область')
        plt.xlabel('Время (с)')
        plt.ylabel('Амплитуда')
        plt.legend()
        plt.grid()
        
        # 4. Спектр и спектрограмма
        plt.subplot(2, 1, 2)
        
        # Вычисление спектрограммы
        n_fft = 2048  # Размер окна для FFT
        hop_length = n_fft // 4  # Шаг между окнами
        stft = librosa.stft(audio_data[:, 0], n_fft=n_fft, hop_length=hop_length)
        spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        
        # Отображение спектрограммы
        librosa.display.specshow(spectrogram, sr=sample_rate, 
                               hop_length=hop_length,
                               x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Спектрограмма (логарифмическая шкала)')
        
        plt.tight_layout()
        plt.show()
        
        # 5. Вывод информации о файле
        print("\nИнформация об аудиофайле:")
        print(f"Путь: {file_path}")
        print(f"Частота дискретизации: {sample_rate} Гц")
        print(f"Длительность: {len(audio_data)/sample_rate:.2f} сек")
        print(f"Форма данных: {audio_data.shape} (отсчеты × каналы)")
        
    except Exception as e:
        print(f"Ошибка при анализе файла: {str(e)}")

if __name__ == "__main__":
    # Путь к аудиофайлу
    audio_file = r'C:\Users\dsorokina\Desktop\Rabota\pulse_signal.wav'
    
    # Проверка существования файла
    if os.path.exists(audio_file):
        analyze_audio_file(audio_file)
    else:
        print(f"Файл не найден: {audio_file}")
        print("Убедитесь, что:")
        print("1. Файл существует по указанному пути")
        print("2. Программа генерации была выполнена ранее")
        print("3. У вас есть права на чтение файла")