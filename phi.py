import time
timestr = time.strftime('%H.%M.%S')
from Kuramoto import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import rcParams
plt.style.use('seaborn-dark-palette')
rcParams['font.family'] = 'serif'
rcParams['mathtext.fontset'] = 'dejavuserif'
import pandas as pd
import os

base_dir = '/home/luke/final_project/phi'

model = Kuramoto(J=0., N=2)

def period(omega_arr):
	''' period of the smallest omega gap  '''
	w, N = sorted(omega_arr), len(omega_arr)
	wDiffs = [w[i + 1] - w[i] for i in range(N-1)]
	deltaW = min(wDiffs)
	if deltaW < 0.01:
		deltaW = 0.01
	return 2*np.pi/deltaW
	
def coupling(strength):
	''' strength = 0 ... uncoupled
		strength < 1 ... unsync expected
		strength = 1 ... sync boundary
		strength > 1 ... sync expected '''
	phi = model.theta0[1] - model.theta0[0]
	J = (model.w[0] - model.w[1])/(2*np.sin(phi))
	return strength*abs(J)
	
stren = 0.905
dur = 50

model.w = np.array([1.1, 0.32])
model.theta0 = np.array([1.95, 3.08])

j = coupling(stren)
model.J = j*model.A
T = period(model.w)
var = [model.w[0], model.w[1], model.theta0[0], model.theta0[1], j]

times, thetas = model.solve(end=dur*T)
phis = np.array([row[1]-row[0] for row in thetas])

header = 'Phase difference evolving through time for initial\n$\omega$ = [{0:.2f}, {1:.2f}] rad/s, $\\theta$ = [{2:.2f}, {3:.2f}] rad at J = {4:.4f}'

fig = plt.figure(1)
plt.title(header.format(*var))
plt.plot(times, phis, 'r-')
#plt.plot([0, times[-1]], [margin, margin], 'b--')
plt.xlabel('t [s]')
plt.ylabel(r'$\phi$ [rad]')
plt.grid(which='both', ls='-', lw=1.)
leg1 = Rectangle((0, 0), 0, 0, alpha=0.0)
leg2 = Rectangle((0, 0), 0, 0, alpha=0.0)
plt.legend([leg1, leg2], ['%.3f $J_{c}$' % stren, '%d periods' % dur], handlelength=0)

print('\n\nPeriod: {:.2f}'.format(T))
print('Initial Omega:\n', model.w)
print('Initial theta:\n', model.theta0)
print('Coupling:\n', np.round(model.J, 2))

fig_name ='phi_%s_temp.png' % timestr
fig.savefig(os.path.join(base_dir, fig_name))

plt.show()

