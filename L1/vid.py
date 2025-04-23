import numpy as np 
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt

Fd = 44100 # частота дискретизации
T = 3 # длительность аудио в сек.
N = round(T*Fd)
def harm_waves(frequencies, amplitudes):
    model_signal = np.zeros((2,N))
    for f, A in zip(frequencies, amplitudes):
        signal_s = A * np.sin(2 * np.pi * f * np.arange(N) / Fd)
        signal_c = A * np.cos(2 * np.pi * f * np.arange(N) / Fd)
        # в левом канале будет из синусов, в правом - из косинусов:
        signal = np.vstack((signal_s, signal_c))
        model_signal = np.add(model_signal, signal)
    return model_signal
frequencies = [200, 400, 600, 800, 1000]
amplitudes = [1, 1/2, 1/3, 1/4, 1/5]
model_signal = harm_waves(frequencies, amplitudes)
# Нормировка – это обязательно всегда!:
Norm = np.max(np.abs(model_signal))
if Norm != 0:
     model_signal = model_signal / Norm
# Строим график сигнала:
start_t, stop_t = 0, 0.02 # границы по времени для визуализации
fig, ax = plt.subplots(nrows=1, sharex=True, figsize=(10, 4))
ax.set(xlim=[start_t, stop_t])
librosa.display.waveshow(model_signal[0, :],
                        sr=Fd, color='b', ax=ax,
                                label='left channel')
librosa.display.waveshow(model_signal[1, :],
                        sr=Fd, color='r', ax=ax,
                                label='right channel')
ax.label_outer()
ax.legend()
ax.grid()
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Audio Signal')
plt.show()
plt.close()
sf.write(r'C:\Users\dsorokina\Desktop\Rabota\zp.mp3', np.transpose(model_signal), Fd)