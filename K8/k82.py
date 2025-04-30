import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import soundfile as sf
import librosa
from scipy import signal
import warnings
from moviepy import VideoFileClip, AudioFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.audio.AudioClip import AudioArrayClip
import matplotlib.pyplot as plt

# Отключаем предупреждения librosa
warnings.filterwarnings("ignore", category=FutureWarning)

def canny_edge_detection(frame):
    """Функция для обработки кадра с помощью алгоритма Канни"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges_rgb

def select_video_file():
    """Открывает диалоговое окно для выбора видеофайла"""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Выберите видеофайл",
        filetypes=[("Видеофайлы", "*.mp4 *.avi *.mov *.mkv"), ("Все файлы", "*.*")]
    )
    return file_path

def extract_audio(video_path, audio_path):
    """Извлечение аудио с помощью moviepy"""
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(audio_path, codec='pcm_s16le')
        video.close()
        return True
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось извлечь аудио: {str(e)}")
        return False

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

def process_audio_with_hpf(audio_path, output_path):
    """Обработка аудио с применением ФВЧ"""
    try:
        # Загрузка аудио
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        
        # Применение ФВЧ
        y_filtered, _, _, _ = fourier_highpass_filter(y, sr)
        
        # Нормализация
        y_filtered = librosa.util.normalize(y_filtered)
        
        # Сохранение результата
        sf.write(output_path, y_filtered, sr, subtype='PCM_16')
        return True
    except Exception as e:
        messagebox.showerror("Ошибка", f"Ошибка обработки аудио: {str(e)}")
        return False

def reverse_audio(audio_clip):
    """Переворачивает аудио задом наперед"""
    audio_array = audio_clip.to_soundarray()
    reversed_array = np.flipud(audio_array)
    return AudioArrayClip(reversed_array, fps=audio_clip.fps)

def process_video():
    """Основная функция обработки видео"""
    input_path = select_video_file()
    if not input_path:
        return

    # Создаем временные файлы
    temp_dir = "temp_audio_video"
    os.makedirs(temp_dir, exist_ok=True)
    temp_audio = os.path.join(temp_dir, "original_audio.wav")
    temp_filtered_audio = os.path.join(temp_dir, "filtered_audio.wav")

    # Извлекаем аудио
    if not extract_audio(input_path, temp_audio):
        return

    # Обрабатываем аудио с ФВЧ
    if not process_audio_with_hpf(temp_audio, temp_filtered_audio):
        return

    # Обрабатываем видео и собираем кадры
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        messagebox.showerror("Ошибка", "Не удалось открыть видеофайл")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = []
    print("Обработка кадров...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = canny_edge_detection(frame)
        frames.append(processed_frame)
    cap.release()

    # Создаем видео клип из кадров
    video_clip = ImageSequenceClip(frames, fps=fps)
    
    # Создаем аудио клип
    audio_clip = AudioFileClip(temp_filtered_audio)
    
    # Применяем эффект переворота аудио
    if messagebox.askyesno("Эффект аудио", "Перевернуть аудио задом наперед?"):
        audio_clip = reverse_audio(audio_clip)
    
    # Комбинируем видео и аудио
    final_clip = video_clip.with_audio(audio_clip)
    
    # Сохраняем результат
    output_path = filedialog.asksaveasfilename(
        title="Сохранить обработанное видео как...",
        defaultextension=".mp4",
        filetypes=[("MP4 файлы", "*.mp4"), ("Все файлы", "*.*")]
    )
    if not output_path:
        return

    final_clip.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac',
        fps=fps,
        threads=4
    )

    # Закрываем клипы
    video_clip.close()
    audio_clip.close()
    final_clip.close()

    # Очистка временных файлов
    for f in [temp_audio, temp_filtered_audio]:
        if os.path.exists(f):
            os.remove(f)
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)

    messagebox.showinfo("Готово", f"Видео с обработанным аудио сохранено: {output_path}")

if __name__ == "__main__":
    process_video()