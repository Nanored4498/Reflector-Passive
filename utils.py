import numpy as np
import pylab as pl
from simul_gauss import simul_1D_gaussian
import time

def sample_ys(N):
	"""
	This function samples N points in [-50,50]x[-10,10]x[185,200]

	Args:
		N (int): number of points to sample

	Output:
		sample (np.array): array of shape (N,3) of the sampled 3D points
	"""
	sample = np.random.uniform(size=(N, 3)) * np.array([100,20,15]) + np.array([-50,-10,185])
	return sample

def recevers_xs():
	"""
	This function creates the receivers x_s

	Args:
		None

	Output:
		sample (np.array): array of shape (5,3) of the receivers
	"""
	sample = np.arange(5)[:,None]*[7.5,0,0] + np.array([-30, 0, 100])
	return sample

def hat_G0(omega, x, y, c_0):
	"""
	This function computes the green function hat_G0

	Args:
		omega (np.array): angular frequency
		x (np.array): inputs of the Green function
		y (np.array): inputs of the Green function
		c_0 (float): celerity

	Output:
		hat_G0 (np.array): value of the function hat G_0
	"""
	X = np.linalg.norm(x - y, axis=-1)
	Exp = np.exp(1.0j * omega * X / c_0)
	hat_G0 = Exp / (4 * np.pi * X)
	return hat_G0

def hat_G(omega, x, y, c_0, z_r, sigma_r):
	"""
	This function computes the green function hat_G

	Args:
		omega (np.array): angular frequency
		x (np.array): inputs of the Green function
		y (np.array): inputs of the Green function
		c_0 (float): velocity
		z_r (np.array): reflector position
		sigma_r (float): multiplicative constant of Taylors second order term

	Output:
		hat_G (np.array): value of the function hat_G
	"""
	hat_G = hat_G0(omega,x,y,c_0) + sigma_r * omega**2 * hat_G0(omega,x,z_r,c_0) * hat_G0(omega,z_r,y,c_0)
	return hat_G

# TODO: Maybe a rescale
def hat_n(omega, N):
	"""
	
	"""
	fC = np.exp(-omega**2)
	Y = np.random.normal(size=(len(omega), N))*(len(omega))**0.5/2 + 0j
	n = len(omega)//2
	Y[:n] *= 1j
	Y[:-n-1:-1] -= Y[:n]
	Y[n-1::-1] += Y[-n:].real
	output = np.sqrt(fC)[:,None] * Y
	return output

def C_N(tau, x_1, x_2, y, c_0, z_r, sigma_r):
	"""
	This function computes the cross correlation of the signals recorded at two
	distinct locations.

	Args:
		tau (float): temporal lag
		x_1 (np.array): position of the first recepter
		x_2 (np.array): position of the second recepter
		y (np.array): position of noise sources
		c_0 (float): celerity
		z_r (np.array): reflector position
		sigma_r (float): multiplicative constant of Taylors second order term

	Output:
		C_N
	"""
	alpha = 4 # Bornes sur omega
	precision = 1000 # nombre de omega
	omega = np.linspace(-alpha, alpha, precision)
	tau2 = tau.flatten()[:,None]
	prodG = np.conj(hat_G(omega[:,None], x_1, y, c_0, z_r, sigma_r)) * hat_G(omega[:,None], x_2, y, c_0, z_r, sigma_r)
	C_N = np.exp(-omega**2) * prodG.mean(1) * np.exp(-1.0j*omega*tau2)
	C_N = C_N.sum(1) * (2*alpha/precision) / (2 * np.pi)
	return C_N.reshape(tau.shape)

def KM(y_S, x, y, c_0, z_r, sigma_r):	
	"""
	This function computes the value of a pixel of the KM image for exercise 1

	Args:
		y_S (np.array): positions where we want to compute the KM image
		x (np.array): positions of recepters
		y (np.array): positions of noise sources
		c_0 (float): velocity
		z_r (np.array): position of the reflector
		sigma_r (float): multiplicative constant of Taylors second order term

	Output:
		I_N(y_S)
	"""
	alpha = 4
	precision = 1000
	yS2 = y_S.reshape(-1, 3)
	dist_yx = np.linalg.norm(yS2[:,None] - x, axis=-1)
	tau = ((dist_yx[:,None] + dist_yx[:,:,None]) / c_0)[:,None]
	omega = np.linspace(-alpha, alpha, precision)[:,None]
	hat_G0_xY = hat_G0(omega[:,None], x[:,None], y, c_0).mean(-1)
	hat_G0_xz = hat_G0(omega, x, z_r, c_0)
	hat_G0_zY = hat_G0(omega, z_r, y, c_0).mean(-1, keepdims=True)
	hat_G_xY = hat_G0_xY + sigma_r * omega**2 * hat_G0_xz * hat_G0_zY
	prodG = np.conj(hat_G_xY[:,:,None]) * hat_G_xY[:,None]
	omega = omega[:,None]
	C_N = np.exp(-omega**2) * prodG * np.exp(-1.0j * omega * tau)
	I = C_N.sum((1, 2, 3)).real * (omega[1]-omega[0]) / (2*np.pi)
	return I.reshape(y_S.shape[:-1])

def C_TNm(tau, x_1, x_2, T, y, c_0, z_r, sigma_r):
	mi_tau = min(0, tau.min()-1)
	DW = 9 # range of omega
	dt = 2*np.pi / DW # time step
	DT = T - mi_tau # range of time
	nt = 1 + int(DT / dt) # number of time steps
	if nt % 2 == 0: nt += 1
	dt = DT / (nt - 1) 
	dw = 2*np.pi / DT # omega step
	omega = np.linspace(-nt//2 * dw , nt//2 * dw, nt)

	n = hat_n(omega, len(y)) * dw
	G1 = hat_G(omega[:,None], x_1, y, c_0, z_r, sigma_r)
	G2 = hat_G(omega[:,None], x_2, y, c_0, z_r, sigma_r)
	Gn1 = (G1 * n).sum(1) / (2*np.pi*np.sqrt(len(y)))	# sum over y
	Gn2 = (G2 * n).sum(1) / (2*np.pi*np.sqrt(len(y)))	# sum over y
	u1 = np.fft.fft(np.fft.ifftshift(Gn1)).real
	u2 = np.fft.fft(np.fft.ifftshift(Gn2)).real
	t0 = int(-mi_tau/dt)
	l0 = -mi_tau/dt - t0

	output = np.empty(tau.shape, float)
	for i in range(len(tau)):
		t1 = int(abs(tau[i])/dt)
		l1 = abs(tau[i])/dt - t1
		t1 += 1
		u1t = u1[t0:nt-t1]
		num_t = nt-t0-t1
		t2 = t0 + int(nt+tau[i]/dt) - nt
		l2 = t0 + tau[i]/dt - t2
		u2t = (1-l2) * u2[t2:t2+num_t] + l2 * u2[t2+1:t2+1+num_t]
		output[i] = ((u1t * u2t).sum() - l0*u1t[0]*u2t[0] - l1*u1t[-1]*u2t[-1]) / (num_t-l0-l1)
	return np.array(output)

def C_TNM(M, tau, x_1, x_2, T, y, c_0, z_r, sigma_r):
	CTNM = 0
	for i in range(M):
		CTNM += C_TNm(tau, x_1, x_2, T, y, c_0, z_r, sigma_r)
	CTNM = CTNM/M
	return CTNM

def etude_resolution(img):
	"""
	We have the same resolution study as for passive imaging (see section 2.3.9 in poly)
	"""
	img = np.power(img, 2)
	R = np.max(img)/np.mean(img)
	return R

def KMT(y_S, x, y, T, M, c_0, z_r, sigma_r):
	DW = 9 # range of omega
	dt = 2*np.pi / DW # time step
	nt = 1 + int(T / dt) # number of time steps
	if nt % 2 == 0: nt += 1
	dt = T / (nt - 1) 
	dw = 2*np.pi / T # omega step
	omega = np.linspace(-nt//2 * dw , nt//2 * dw, nt)

	yS2 = y_S.reshape(-1, 3)
	I = np.zeros(yS2.shape[0])
	dist_xy = np.linalg.norm(yS2[:,None] - x, axis=-1).T
	tau = ((dist_xy[None] + dist_xy[:,None]) / c_0)
	Gxwy = hat_G(omega[:,None,None], x[:,None], y, c_0, z_r, sigma_r).transpose(1, 0, 2)

	for m in range(M):
		print(f'\rRealization: {m+1}/{M}', end='')
		nwy = hat_n(omega, len(y)) * dw
		Gnxw = (Gxwy * nwy).sum(2) / (2*np.pi*np.sqrt(len(y)))	# sum over y
		u = np.fft.fft(np.fft.ifftshift(Gnxw))
		for i in range(len(x)):
			for j in range(len(x)):
				for k in range(len(yS2)):
					tau_k = tau[i,j,k]
					t1 = int(tau_k/dt)
					l1 = tau_k/dt - t1
					num_t = nt-t1-1
					u1t = u[i,:num_t]
					u2t = (1-l1) * u[j,t1:t1+num_t] + l1 * u[j,t1+1:t1+1+num_t]
					I[k] += ((u1t * u2t).sum() - l1*u1t[-1]*u2t[-1]) / (num_t-l1)
	
	print('', end='\r')
	return I.reshape(y_S.shape[:-1])