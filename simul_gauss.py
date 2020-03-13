import numpy as np
# import pylab as pl

def simul_1D_gaussian(n=1025, h=0.05):
	omega = np.arange(-int(n/2), int(n/2)+1) * h
	C = np.exp(-omega**2)
	Y = np.random.normal(loc=0.0, scale=1.0, size=n)
	tmp = np.fft.ifftshift(C)
	hat_f = np.sqrt(tmp) * np.fft.fft(Y)
	return hat_f

# n = 1025
# h = 0.01
# hf = simul_1D_gaussian(n, h)
# f = np.fft.ifft(hf).real
# f -= f.mean()
# cf = np.array([(np.roll(f, i)*f).mean() for i in range(n)])
# hat_cf = np.fft.fftshift(abs(np.fft.fft(cf)))
# omega = np.arange(-int(n/2), int(n/2)) * h
# C = np.exp(-omega**2)
# pl.plot(hat_cf)
# pl.plot(C)
# pl.show()
# print(hat_cf[n//2+1], cf.sum())
# print(hat_cf.min(), hat_cf.max())
# pl.plot((-np.log(hat_cf))**0.5)
# pl.show()