import sys
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import chirp
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian

Fd = 44100 # частота дискретизации
T = 20 # длительность аудио в сек.
N = round(T*Fd)
t = np.linspace(0, T, N)
# ЛЧМ - за Т секунд частота возрастет с 300 до 3000 Гц:
model_chirp = chirp(t, f0=300, f1=3000, t1=T, method='linear')
# сформируем стереосигнал:
model_signal = np.vstack((model_chirp, model_chirp))
Norm = np.max(np.abs(model_signal))
if Norm != 0:
 model_signal = model_signal / np.max(np.abs(model_signal))
plt.close('all') # Очистка памяти – это не лишне
# вычисляем амплитудный спектр сигнала (левый канал):
Spectr_input = np.fft.fft(model_signal[0,:]) # БПФ
# Преобразуем в дБ:
AS_input = np.abs(Spectr_input) # взяли модуль
eps = np.max(AS_input) * 1.0e-9 # чтобы избежать lg(0)
S_dB_input = 20 * np.log10(AS_input + eps) # спектр в дБ
# Строим график амплитудного спектра:
f = np.arange(0, Fd/2, Fd/N) # набор частот
S_dB_input = S_dB_input[:len(f)] # выравниваем длины
plt.figure(figsize=(6, 4))
plt.semilogx(f, S_dB_input) # график в полулог. масштабе!!
plt.grid(True)
plt.minorticks_on() # отобразит мелкую сетку на лог.масштабе
plt.grid(True, which='major', color = '#444', linewidth = 1)
plt.grid(True, which='minor', color='#aaa', ls=':')
# зададим лимиты для осей, по вертикали - кратно 20:
Max_dB = np.ceil(np.max(S_dB_input)/20)*20
plt.axis([10, Fd/2, Max_dB-100, Max_dB]) # ограничим оси
plt.xlabel('Frequency (Hz)')
plt.ylabel('Level (dB)')
plt.title('Amplitude Spectrum')
plt.show()

# Вычисляем с строим спектрограмму – переработано с [3]:
# standard deviation for Gauss. window (in samples!):
g_std = 0.2*Fd
# symmetric Gaussian window:
wind = gaussian(round(2*g_std), std=g_std, sym=True)
SFT = ShortTimeFFT(wind,
 hop=round(0.1*Fd), # прыгаем по 0.1 сек
 fs=Fd,
scale_to='magnitude')
Sx = SFT.stft(model_signal[0,:]) # perform the STFT
print(sys.getsizeof(Sx)) #это чтобы знать сколько памяти съели
# далее только построение графика спектрограммы:
fig1, ax1 = plt.subplots(figsize=(6, 4))
t_lo, t_hi = SFT.extent(N)[:2] # time range of plot
# тут очень хитро сделанные надписи на осях, жаль чистить:
ax1.set_title(rf"STFT ({SFT.m_num*SFT.T:g}$\,s$ Gauss window,"+
 rf"$\sigma_t={g_std*SFT.T}\,$s)")
ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices,"+
 rf"$\Delta t = {SFT.delta_t:g}\,$s)",
 ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, "+
 rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
 xlim=(t_lo, t_hi))
epss=np.max(abs(Sx))*1e-6
im1 = ax1.imshow(20*np.log10(abs(Sx)+epss),
 origin='lower', aspect='auto',
 extent=SFT.extent(N), cmap='viridis')
fig1.colorbar(im1, label="Magnitude $|S_x(t, f)|, dB $")
#im1 = ax1.imshow(abs(Sx), origin='lower', aspect='auto',
#extent=SFT.extent(N), cmap='viridis')
fig1.colorbar(im1, label="Magnitude $|S_x(t, f)|$")
ax1.semilogy()
ax1.set_xlim([0, T])
ax1.set_ylim([10, Fd/2])
# Show the major grid and style it slightly.
ax1.grid(which='major', color='#bbbbbb', linewidth=0.5)
# Show the minor grid as well.
# Style it in very light gray as a thin, dotted line.
ax1.grid(which='minor', color='#999999', linestyle=':',
linewidth=0.5)
# Make the minor ticks and gridlines show.
ax1.minorticks_on()
plt.show()


real_signal, sr_real = librosa.load('zp.mp3', sr=None, mono=False)
print(f"Длина сигнала: {len(real_signal)} отсчетов")
print(f"Частота дискретизации: {sr_real} Гц")
print(f"Длительность: {len(real_signal)/sr_real:.2f} сек")