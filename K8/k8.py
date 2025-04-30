import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import librosa
from scipy import signal
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.audio.AudioClip import AudioArrayClip

# Отключаем предупреждения librosa
import warnings
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
        return audio
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось извлечь аудио: {str(e)}")
        return None

def generate_water_sound(duration=1.0, sr=44100):
    """Генерация искусственного шума воды"""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Основные частоты для звука воды (100-1000 Гц)
    base_freq = 100 + 900 * np.random.random()
    
    # Генерация белого шума
    noise = np.random.normal(0, 0.5, len(t))
    
    # Применяем фильтр для имитации звука воды
    b, a = signal.butter(4, [base_freq/(sr/2), (base_freq+300)/(sr/2)], btype='bandpass')
    water_sound = signal.lfilter(b, a, noise)
    
    # Добавляем "плескающие" эффекты
    for _ in range(3):
        freq = 50 + 500 * np.random.random()
        env = np.exp(-5 * t)  # Экспоненциальная огибающая
        water_sound += 0.3 * env * np.sin(2 * np.pi * freq * t)
    
    # Нормализация
    water_sound = 0.5 * water_sound / np.max(np.abs(water_sound))
    return np.vstack([water_sound, water_sound]).T  # Преобразуем в стерео

def apply_water_sound(audio_clip, interval=3.0, water_duration=1.0):
    """Добавляет периодический шум воды к аудио"""
    try:
        sr = audio_clip.fps
        duration = audio_clip.duration
        
        # Создаем список аудиоклипов
        clips = [audio_clip]
        
        # Добавляем шум воды через заданные интервалы
        for start_time in np.arange(0, duration, interval):
            if start_time + water_duration > duration:
                break
                
            # Генерируем шум воды
            water_array = generate_water_sound(water_duration, sr)
            water_clip = AudioArrayClip(water_array, fps=sr)
            water_clip = water_clip.set_start(start_time)
            clips.append(water_clip)
        
        # Объединяем все клипы
        final_audio = CompositeAudioClip(clips)
        return final_audio
    except Exception as e:
        messagebox.showerror("Ошибка", f"Ошибка добавления шума воды: {str(e)}")
        return audio_clip

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
    original_audio = extract_audio(input_path, temp_audio)
    if original_audio is None:
        return

    # Добавляем периодический шум воды
    water_audio = apply_water_sound(original_audio, interval=3.0, water_duration=1.0)
    
    # Сохраняем временный аудиофайл
    water_audio.write_audiofile(temp_filtered_audio, codec='pcm_s16le', ffmpeg_params=["-ac", "2"])

    # Обрабатываем видео и собираем кадры
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        messagebox.showerror("Ошибка", "Не удалось открыть видеофайл")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    print("Обработка кадров...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = canny_edge_detection(frame)
        frames.append(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
    cap.release()

    # Создаем видео клип из кадров
    video_clip = ImageSequenceClip(frames, fps=fps)
    
    # Создаем аудио клип с добавленным шумом воды
    audio_clip = AudioFileClip(temp_filtered_audio)
    
    # Комбинируем видео и аудио
    final_clip = video_clip.set_audio(audio_clip)
    
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
        threads=4,
        preset='fast',
        bitrate='3000k'
    )

    # Закрываем клипы
    video_clip.close()
    audio_clip.close()
    final_clip.close()
    original_audio.close()

    # Очистка временных файлов
    for f in [temp_audio, temp_filtered_audio]:
        if os.path.exists(f):
            os.remove(f)
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)

    messagebox.showinfo("Готово", f"Видео с периодическим шумом воды сохранено: {output_path}")

if __name__ == "__main__":
    process_video()