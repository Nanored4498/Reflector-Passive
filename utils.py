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
	fC = np.fft.ifftshift(np.exp(-omega**2))
	Y = np.random.normal(loc=0.0, scale=1.0, size=(N, *omega.shape))
	output = np.sqrt(fC) * np.fft.fft(Y)
	return output.T

def u(t, omega, n, x, y, c_0, z_r, sigma_r):
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
	assert n.shape[1] == y.shape[0]
	G = hat_G(omega[:,None], x, y, c_0, z_r, sigma_r)
	Gn = (G * n).sum(1)	# sum over y
	t = np.expand_dims(t, -1)
	u_tx = (Gn * np.exp(-1.0j*omega*t)).sum(-1) * (omega[1]-omega[0]) # integral over omega
	u_tx /= (2*np.pi*np.sqrt(len(y)))
	return u_tx

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
	alpha_omega = 6
	precision_omega = 500
	precision_t = 800
	omega = np.linspace(-alpha_omega, alpha_omega, precision_omega)

	n = hat_n(omega, len(y))
	G1 = hat_G(omega[:,None], x_1, y, c_0, z_r, sigma_r)
	G2 = hat_G(omega[:,None], x_2, y, c_0, z_r, sigma_r)
	Gn1 = (G1 * n).sum(1) / (2*np.pi*np.sqrt(len(y)))	# sum over y
	Gn1 *= omega[1]-omega[0] # step size of the integration over omega
	Gn2 = (G2 * n).sum(1) / (2*np.pi*np.sqrt(len(y)))	# sum over y
	Gn2 *= omega[1]-omega[0] # step size of the integration over omega

	output = []
	for tt in tau:
		print(tt)
		t = np.linspace(0, T - tt, precision_t)[:,None]
		u_1 = (Gn1 * np.exp(-1.0j*omega*t)).sum(-1)
		u_2 = (Gn2 * np.exp(-1.0j*omega*(t + tt))).sum(-1)
		output.append((u_1 * u_2).mean())
	return np.array(output)

def C_TNM(M, tau, x_1, x_2, T, y, c_0, z_r, sigma_r):
	CTNM = C_TNm(tau, x_1, x_2, T, y, c_0, z_r, sigma_r)
	for i in range(M-1):
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







