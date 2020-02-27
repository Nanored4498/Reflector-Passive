import utils as utils
import numpy as np
import matplotlib.pyplot as plt

questions_ready = [
	[1, 1, 1],
	[1, 0, 0]
]

def exercice_1(q1, q2, q3):
	N = 100
	w_size = 10

	print("Resolution de l'exercice 1 : ")
	print("On considere le cas T -> +inf")

	if q1 == 1:
		print("\ta) on plot tau -> C_N(tau,x_1,x_5) sur l'intervalle [-200;200]")

		tau = np.linspace(-200,200, 1000)[:,None,None]
		x_1 = utils.x[0]
		x_5 = utils.x[-1]
		y = utils.sample_ys(N)
		f = utils.C_N(tau, x_1, x_5, y, utils.c_0, utils.z_r, utils.sigma_r)
		plt.plot(tau.flatten(), f, 'r-')
		plt.xlabel(r'$\tau$')
		plt.ylabel(r'$C_N$')
		plt.title(r'$\tau \rightarrow C_N(\tau,x_1,x_5)$')
		plt.savefig('figs/plot tau -> C_N(tau,x_1,x_5).png')
		plt.close()
	elif q1 == 0:
		print("\ta) ---skip---")
	else:
		print("\ta) déjà fait")
	if q2 == 1:
		print("\tb) on plot l'image KM (voir enonce pour la definition)")
		Im = np.zeros((2 * w_size + 1, 2 * w_size + 1))
		y = utils.sample_ys(N)
		for i in range(-w_size, w_size + 1):
			for j in range(-w_size, w_size + 1):
				if j == 0 and i%2 == 0:
					print("\t\tcalcul en cours : " + str(int(100 * float(w_size + i ) / (2 * w_size + 1))) + '%')
				y_s = utils.z_r + np.array([i,0,j])
				Im[i + w_size, j + w_size] = utils.KM(y_s, utils.x, y, utils.c_0, utils.z_r, utils.sigma_r)
		plt.imshow(Im)
		plt.savefig("figs/image KM de I_N.png")
	elif q2 == 0:
		print("\tb) ---skip---")
	else:
		print("\tb) déjà fait")
	if q3 == 1:
		print("\tc) On étudie les proprietes de resolution de l'image (voir rapport)")
	elif q3 == 0:
		print("\tc) ---skip---")
	else:
		print("\tc) déjà fait")
	print("\n")
def exercice_2(q1 = True, q2 = True, q3 = True):
	N = 100

	print("Resolution de l'exercice 2 : ")
	print("On considere le cas T < +inf")
	if q1 == 1:
		print("\ta) on plot tau -> C_NT(tau,x_5,x_1) sur l'intervalle [-150;150] pour différentes valeurs de T")
		x = utils.x
		y = utils.sample_ys(N)
		z_r = utils.z_r
		c_0 = utils.c_0
		sigma_r = utils.sigma_r
		tau_values = np.linspace(-150,150,200)
		T_values = [500, 1000, 10000]
		for T in T_values:
			f = [utils.C_TNm(tau = tau, x_1 = x[0], x_2 = x[1], T = T, y = y, c_0 = c_0, z_r = z_r, sigma_r = sigma_r) for tau in tau_values]
			plt.plot(tau_values, f, 'r-')
			plt.xlabel(r'$\tau$')
			plt.ylabel(r'$C_N$')
			plt.title(r'$\tau \rightarrow C_{N,T}(\tau,x_5,x_1)$')
			plt.savefig('figs/plot tau -> C_NT(tau,x_5,x_1) ' + str(T) + '.png')
			plt.close()
		print("on va maintenant calculer la moyenne des cross-correlations ")
		f = [utils.C_TNM(M = 20, tau = tau, x_1 = x[0], x_2 = x[1], T = 500, y = y, c_0 = c_0, z_r = z_r, sigma_r = sigma_r) for tau in tau_values]
		plt.plot(tau_values, f, 'r-')
		plt.xlabel(r'$\tau$')
		plt.ylabel(r'$C_N$')
		plt.title(r'$\tau \rightarrow C_{N,T,M}(\tau,x_5,x_1)$')
		plt.savefig('figs/plot tau -> C_NTM(tau,x_5,x_1) 500.png')
		plt.close()
	elif q1 == 0:
		print("\ta) ---skip---")
	else:
		print("\ta) déjà fait")
	if q2 == 1:
		print("\tb)")
	elif q2 == 0:
		print("\tb) ---skip---")
	else:
		print("\tb) déjà fait")
	if q3 == 1:
		print("\tc)")
	elif q3 == 0:
		print("\tc) ---skip---")
	else:
		print("\tc) déjà fait")






