import numpy as np
import pylab as pl

domain = np.array([[-50, 50], [-10, 10], [185, 200]])
N = 200
y = np.random.uniform(size=(N, 3)) * (domain[:,1] - domain[:,0]) + domain[:,0]
x = [-30, 0, 100] + np.arange(5)[:,None]*[7.5, 0, 0]
c0 = 1
zr = np.array([-5, 0, 65])
sigmar = 1e-3

showFig = False
if showFig:
	pl.scatter(x[:,2], x[:,0], marker='^')
	pl.scatter(y[:,2], y[:,0], marker='o', facecolor='none', edgecolors='r')
	pl.scatter(zr[2], zr[0], marker='d')
	pl.xlabel("z")
	pl.ylabel("x")
	pl.xlim(0, 200)
	pl.show()

fF = lambda w: np.exp(-w**2)
def fG0(w, x, y):
	dist = np.linalg.norm(x - y, axis=-1)
	return np.exp(1.0j * w * dist / c0) / (4*np.pi * dist)
fG = lambda w, x, y: fG0(w, x, y) + sigmar * w**2 * fG0(w, x, zr) * fG0(w, zr, y)

def CN(tau, i, j):
	Nw = 1000
	w = np.linspace(-4, 4, Nw)[:,None]
	f = fF(w) * np.conj(fG(w, x[i], y)) * fG(w, x[j], y) * np.exp(-1.0j*w*tau)
	res = f.sum((-1,-2)) / (2*np.pi*N) * 8/Nw
	return res

# tau = np.linspace(-200, 200, 500)[:,None,None]
# cn = CN(tau, 4, 0)
# pl.plot(tau.flatten(), cn)
# pl.show()

def KM(ys):
	ds = np.linalg.norm(ys - x, axis=-1)
	n = len(x)
	i = np.arange(n)[:,None,None,None]
	j = np.arange(n)[:,None,None]
	tau = ds[i] - ds[j]
	return CN(tau, i, j).sum()

size = 10
I = np.zeros((2*size+1, 2*size+1))
for i in range(-size, size+1):
	for j in range(-size, size+1):
		print(i, j)
		I[i+size,j+size] = KM(zr + [i, 0, j])
print(I)
pl.imshow(I)
pl.show()