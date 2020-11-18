import time
start = time.time()
from Kuramoto import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.style.use('seaborn-dark-palette')

model = Kuramoto(0.) # choose omegas

def coupling(strength):
	''' strength = 0 ... uncoupled
		strength < 1 ... unsync expected
		strength = 1 ... sync boundary
		strength > 1 ... sync expected '''
	phi = model.theta0[1] - model.theta0[0] 
	return strength*abs(model.w[0] - model.w[1]/(2*np.sin(phi)))
	
def period(omega_arr):
	omega = np.amin(abs(omega_arr)) # slower node has longer period
	return 2*np.pi/omega
	
stren = 2.
T = period(model.w)

j = coupling(stren)
model.J = j*model.A
times, thetas = model.solve(T)

theta_dots = np.array([])
for theta in thetas: 
	theta_dot = model(theta) 
	theta_dots = np.append(theta_dots, theta_dot) 

theta_dots = np.reshape(theta_dots, (thetas.shape))
phi_list = np.array([abs(row[1]-row[0]) for row in theta_dots])

header1='Frequency difference evolving through time for initial\n$\omega$ = [{0:.2f}, {1:.2f}] rad/s, \
$\\theta$ = [{2:.2f}, {3:.2f}] rad at J = {4:.2f}'
var = [model.w[0], model.w[1], model.theta0[0], model.theta0[1], j]

marg = 5e-3

plt.figure(1)
plt.title(header1.format(*var))
plt.plot(times, phi_list, label='One period')
plt.plot([0, times[-1]], [marg, marg], 'r-')#, label='{:.0f}% strength'.format(stren*100))
plt.xlabel('t [s]')
plt.ylabel(r'd$\phi$/dt [rad/s]')
plt.grid(which='both', ls='-', lw=1.)
leg1 = Rectangle((0, 0), 0, 0, alpha=0.0)
leg2 = Rectangle((0, 0), 0, 0, alpha=0.0)
plt.legend([leg1, leg2], ['{:.0f}% strength'.format(stren*100), 'Over 1 period'], handlelength=0)

Margincount = 0
for phi in phi_list:
	if phi < marg:
		Margincount += 1

print('\nmargin count: ', Margincount,'/', len(phi_list),'\n')

syncCount = 0
if phi_list[-1] <= marg:
	print('Sync')
	syncCount += 1

if Margincount == len(phi_list):
	f = open('error.csv', 'ab')
	np.savetxt(f, np.c_[var])
	f.close()

np.savetxt('params_temp.csv', var, delimiter=",")
print('\nRuntime: {0:0.1f} seconds'.format(time.time() - start),'\n')
plt.show()

