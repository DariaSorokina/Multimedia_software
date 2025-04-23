import numpy as np
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt

Fd = 44100 # частота дискретизации
T = 3 # длительность аудио в сек.
N = round(T*Fd)
def harm_wave(f, A, N): # M -длина фрагмента
 signal_s = A * np.sin(2 * np.pi * f * np.arange(N) / Fd)
 signal_c = A * np.cos(2 * np.pi * f * np.arange(N) / Fd)
 model_signal = np.vstack((signal_s, signal_c))
 return model_signal
f, A= 432, 1.0
model_harm = harm_wave(f, A, N)
# огибающая амплитуды:
w = 0.25*(1 - np.cos(2 * np.pi * np.arange(N) / (Fd * T)))**2
# поэлементное умножение массивов:
model_signal = model_harm * w