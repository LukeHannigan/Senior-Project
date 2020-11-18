import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import rcParams
plt.style.use('seaborn-dark-palette')
rcParams['font.family'] = 'serif'
import numpy as np
import pandas as pd
import os

base_dir = '/home/luke/final_project/convergence'

marg_list = [5e-1, 5e-2, 5e-3, 5e-4, 5e-5]
dur = 100
trials = 200
N = 2
var = [N, trials, dur]
cols = ['y', 'b', 'c', 'm', 'g', 'r']
data_root = 'convMargin_{0}N{1}T{2}'.format(*var)

header = 'P(sync) against coupling strength \
for {0} oscillators over\n{2} periods and {1} trials with varied sync threshold'
fig = plt.figure(1)
ax = plt.subplot(111)
ax.set_title(header.format(*var))
ax.set_xlabel('Coupling J')
ax.set_ylabel('P(sync)')

for i, margin in enumerate(marg_list):
	data_name = data_root + ('_%.0E.csv' % margin)
	frame = pd.read_csv(os.path.join(base_dir, data_name))
	frame = frame.truncate(0, 14) # cut converge pts
	data_arr = frame.to_numpy()

	j_list = np.array([row[2] for row in data_arr])
	j_list = j_list#*6. # to give net coupling
	prob_list = np.array([row[0] for row in data_arr])
	p_err = np.array([row[1] for row in data_arr])
	
	label = '%.0E threshold' % margin
	
	ax.errorbar(j_list, prob_list, yerr=p_err, fmt='%s.-'%cols[i],  ecolor=cols[i], elinewidth=1.5, label=label)
	
def critCoupling(pop): # critical coupling for given population N
	deg = pop - 1 # degree for a global network
	critFormula = 2*np.sqrt(2*np.pi)/np.pi
	return (deg/pop)*critFormula
	
Jc = critCoupling(N)
print('Critical Coupling: ', Jc)

ax.legend(loc=0)
ax.grid(which='both', ls='-', lw=1.)
ax.set_ylim(0., 1.05)
ax.set_xlim(0., j_list[-1])
ax.plot([Jc, Jc], [0, 1.5], color='black', ls='-', lw=1.5)
ax.text(Jc+0.03, +0.05, '$J_{c}$')

''' convMargin_4N200T100.eps | 4 node global network over 200 trials and 100 periods '''
fig_name ='convMargin_{0}N{1}T{2}_temp.png'.format(*var)
fig.savefig(os.path.join(base_dir, fig_name))

plt.show()
	
	
