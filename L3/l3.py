import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy import signal
import soundfile as sf
plt.close('all') # Очистка памяти
# Загрузка данных звукового файла - стерео
input_signal, Fd = librosa.load(r'C:/Users/dsorokina/Desktop/Rabota/L3/br.mp3', sr=None,
mono=False)
# Получить длину данных аудиофайла
N = len(np.transpose(input_signal))
T = round(N / Fd)
t = np.linspace(0, T, N)
# ------------------------------------------------------------
# Задаем граничные частоты полосы пропускания фильтра,в Герцах
lower_frequency = 1500
upper_frequency = 2000
# для фильтров высоких порядков расчет нужно осуществлять
# для реализации в каскадной форме ур-ниями 2-го порядка!
'''
 B0(z) B1(z) B{n-1}(z)
H(z) = ----- * ----- * ... * ---------
 A0(z) A1(z) A{n-1}(z)
'''
# типы фильтров: ‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’
# здесь bandpass. Ключ sos означает расчет в каскадной форме
order = 2 # порядок фильтра-прототипа


#- для расчета широкополосного режекторного фильтра с частотами среза
#lower_frequency и upper_frequency:
sos = signal.ellip(order,
 rp = 0.5, rs = 100.0,
 Wn = (lower_frequency / (Fd/2),
 upper_frequency / (Fd/2)),
 btype='bandpass', output='sos')


# метод .butter(...) может быть заменен на:
# .cheby1(order, rp, Wn, btype='bandpass', output='sos')
# где rp - неравномерность АЧХ в дБ. Напр., rp =0.5
# .bessel(order, Wn, btype='bandpass', output='sos')
# .ellip(order, rp, rs, Wn, btype='bandpass', output='sos')
# где rp и rs - неравномерности АЧХ в полосах
# пропускания и задерживания, напр., rp=0.5, rs=0.5
# а теперь собственно, фильтрация:
output_signal = signal.sosfilt(sos, input_signal) # готово
# ----------------------------------------------------------
# Расчет амплитудно-частотной хар-ки по коэф-ам фильтра (sos)
f, H = (signal.sosfreqz(sos, worN=Fd, whole=False, fs=Fd))
eps = 1e-10 # чтобы избежать lg(0) при переводе в дБ
L = 20 * np.log10(abs(H)+eps) # перевод в дБ
# Построим график АЧХ фильтра, чтобы знать
# каков его коэффициент пропускания на разных частотах:
plt.semilogx(f, L)
plt.title('Digital filter frequency response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Level [dB]')
plt.xlim(10, Fd/2) # limit x axis
plt.ylim(-80, 20) # limit y axis
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
# отметки граничных частот:
plt.axvline(lower_frequency, color='green')
plt.axvline(upper_frequency, color='green')
plt.show()
# Полезно сравнить спектры до и после фильтрации
# построив их на одном графике:
# вычисляем спектр входного сигнала
Spectr_input = np.fft.fft(input_signal)
# Преобразуем в дБ:
AS_input = np.abs(Spectr_input)
eps = np.max(AS_input) * 1.0e-9
S_dB_input = 20 * np.log10(AS_input + eps)

Spectr_output_real = np.fft.fft(output_signal)
S_dB_output_real = 20 * np.log10(np.abs(Spectr_output_real+eps))
f = np.arange(0, Fd/2, Fd/N) # Перевести Абсциссу в Гц
S_dB_output_real = S_dB_output_real[:, :len(f)]
S_dB_input = S_dB_input[:, :len(f)]


plt.figure(figsize=(6, 4))
plt.semilogx(f, S_dB_input[0, :], color='b',
 label=r'input spectrum')
plt.semilogx(f, S_dB_output_real[0, :], color='r',
 label=r'output spectrum')
plt.grid(True)
plt.minorticks_on() # отобразит мелкую сетку на лог.масштабе
plt.grid(True, which='major', color = '#444', linewidth = 1)
plt.grid(True, which='minor', color='#aaa', ls=':')
# делаем красивый автомасштаб на оси ординат:
Max_A = np.max((np.max(np.abs(Spectr_input)),
 np.max(np.abs(Spectr_output_real))))
Max_dB = np.ceil(np.log10(Max_A))*20
plt.axis([10, Fd/2, Max_dB-120, Max_dB])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Level (dB)')
plt.title('Amplitude Spectrums of input and output audio')
plt.legend()
plt.show()
# Выводим графики исходного аудиосигнала и после фильтрации:
start_t, stop_t = 0, T
plt.figure(figsize=(6, 3))
plt.subplot(2, 1, 1)
plt.plot(t, input_signal[0, :])
plt.xlim([start_t, stop_t])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Input Audio Signal')
plt.subplot(2, 1, 2)
plt.plot(t, output_signal[0, :])
plt.xlim([start_t, stop_t])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Output Audio Signal')
plt.tight_layout()
plt.show()
# Записываем новый аудиофайл
sf.write(r'C:/Users/dsorokina/Desktop/Rabota/L3/audio2.mp3', np.transpose(output_signal), Fd)