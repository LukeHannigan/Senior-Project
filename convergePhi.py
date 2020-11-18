import time
start = time.time()
from Kuramoto import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.style.use('seaborn-dark-palette')

model = Kuramoto(0.)

def coupling(strength):
	''' strength = 0 ... uncoupled
		strength < 1 ... unsync expected
		strength = 1 ... sync boundary
		strength > 1 ... sync expected '''
	phi = model.theta0[1] - model.theta0[0]
	return strength*abs(model.w[0] - model.w[1]/(2*np.sin(phi)))
	
def period(omega_arr):
	''' period of the slowest node '''
	omega = np.amin(abs(omega_arr))
	return 2*np.pi/omega

stren = 10.

j = coupling(stren)
model.J = j*model.A
T = period(model.w)
times, thetas = model.solve(T)

phis = np.array([row[1]-row[0] for row in thetas])

header = 'Phase difference evolving through time for initial\n$\omega$ = [{0:.2f}, {1:.2f}] rad/s, $\\theta$ = [{2:.2f}, {3:.2f}] rad at J = {4:.2f}'
var = [model.w[0], model.w[1], model.theta0[0], model.theta0[1], j]

plt.figure(1)
plt.title(header.format(*var))
plt.plot(times, phis, 'r-')
plt.plot([0, times[-1]], [margin, margin], 'b--')
plt.xlabel('t [s]')
plt.ylabel(r'$\phi$ [rad]')
plt.grid(which='both', ls='-', lw=1.)
leg1 = Rectangle((0, 0), 0, 0, alpha=0.0)
leg2 = Rectangle((0, 0), 0, 0, alpha=0.0)
plt.legend([leg1, leg2], ['{:.0f}% strength'.format(stren*100), 'Over 1 period'], handlelength=0)

print('\nStrength: {:.0f} %'.format(stren*100.))
print('Period: {:.2f}'.format(T))
print('\nRuntime: {0:0.1f} seconds'.format(time.time() - start),'\n')

plt.show()

