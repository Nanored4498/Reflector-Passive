import numpy as np
from simul_gauss import simul_1D_gaussian

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
		omega (np.array): inputs of the Green function
		x (np.array): inputs of the Green function
		y (np.array): inputs of the Green function
		c_0 (float): medium caracteristic

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
		omega (np.array): inputs of the Green function
		x (np.array): inputs of the Green function
		y (np.array): inputs of the Green function
		c_0 (float): medium caracteristic
		z_r (np.array): intermidiate point
		sigma_r (float): multiplicative constant of Taylors second order term

	Output:
		hat_G (np.array): value of the function hat G_
	"""

	hat_G = hat_G0(omega,x,y,c_0) + sigma_r * omega **2 * hat_G0(omega,x,z_r,c_0) * hat_G0(omega,z_r,y,c_0)
	return hat_G

x = recevers_xs()
c_0 = 1
z_r = np.array([65,0,65])
sigma_r = 10**(-3)

def hat_n(omega):
	"""
	
	"""

	output = simul_1D_gaussian(n = len(omega), h = np.abs(omega[0] - omega[1]))
	return output
def u(t, x, y, c_0, z_r, sigma_r):
	"""
	This function computes the recorded signal at x in time t

	Args:
		t (float): recording time
		x (np.array): inputs of the Green function
		y (np.array): inputs of the Green function
		c_0 (float): medium caracteristic
		z_r (np.array): intermidiate point
		sigma_r (float): multiplicative constant of Taylors second order term


	Output:
		u_tx
	"""

	alpha = 10
	precision = 1000
	omega = np.linspace(-alpha, alpha, precision)[:,None]
	G = hat_G(omega, x, y, c_0, z_r, sigma_r)
	n = hat_n(omega)[:,None]
	e = np.exp(-1.0j*omega*t)
	u_tx = G * n * e
	u_tx = u_tx.sum(axis = -1)/(2*np.pi*np.sqrt(len(y)))
	return u_tx
def C_N(tau, x_1, x_2, y, c_0, z_r, sigma_r):
	"""
	This function computes the recorded signal at x in time t

	Args:
		tau (float): ???
		x_1 (np.array): 3D point to evaluate C_N
		x_2 (np.array): 3D point to evaluate C_N
		y (np.array): inputs of the Green function
		c_0 (float): medium caracteristic
		z_r (np.array): intermidiate point
		sigma_r (float): multiplicative constant of Taylors second order term


	Output:
		C_N
	"""
	alpha = 10
	precision = 1000
	omega = np.linspace(-alpha, alpha, precision)[:,None]
	C_N = np.exp(-omega**2) * np.conj(hat_G(omega, x_1, y, c_0, z_r, sigma_r)) * hat_G(omega, x_2, y, c_0, z_r, sigma_r) * np.exp(-1.0j*omega*tau)
	C_N = C_N.sum((-1,-2)) * ((2*alpha)/precision) * (1 / (2 * np.pi * len(y)))
	return C_N
def KM(y_S, x, y, c_0, z_r, sigma_r):
	"""
	This function computes the value of a pixel of the KM image for exercise 1

	Args:
		y_S (np.array): inputs of the C_N function
		x (np.array): inputs of the C_N function
		y (np.array): inputs of the C_N. function
		c_0 (float): medium caracteristic
		z_r (np.array): intermidiate point
		sigma_r (float): multiplicative constant of Taylors second order term


	Output:
		output
	"""

	output = 0
	for i in range(len(x)):
		for j in range(len(x)):
			tau = np.linalg.norm(x[i] - y_S) + np.linalg.norm(x[j] - y_S)
			output += C_N(tau, x[i], x[j], y, c_0, z_r, sigma_r)
	return output
def C_TNm(tau, x_1, x_2, T, y, c_0, z_r, sigma_r):
	alpha = 10
	precision = 1000
	t = np.linspace(-alpha, alpha, precision)[:,None]
	u_1 = u(t = t, x = x_1, y = y, c_0 = c_0, z_r = z_r, sigma_r = sigma_r)
	u_2 = u(t = t + tau, x = x_2, y = y, c_0 = c_0, z_r = z_r, sigma_r = sigma_r)
	C_TNm = u_1.sum(axis = -1) + u_2.sum(axis = -1)
	C_TNm = C_TNm/(T - np.abs(tau))
	return C_TNm
def C_TNM(M, tau, x_1, x_2, T, y, c_0, z_r, sigma_r):
	CTNM = C_TNm(tau, x_1, x_2, T, y, c_0, z_r, sigma_r)
	for i in range(M-1):
		CTNM += C_TNm(tau, x_1, x_2, T, y, c_0, z_r, sigma_r)
	CTNM = CTNM/M
	return CTNM









