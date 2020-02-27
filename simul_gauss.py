import numpy as np

def simul_1D_gaussian(n = 2**10, h = 0.05):
	X = np.arange(-int(n/2), int(n/2)) * h
	C = np.exp(-X**2)
	Y = np.random.normal(loc=0.0, scale=1.0, size=n)
	Yp = np.diff(Y)
	tmp = np.fft.fft(np.fft.fftshift(C))
	hat_f = np.sqrt(tmp) * np.fft.fft(Y)
	return X, np.real(np.fft.ifft(hat_f))