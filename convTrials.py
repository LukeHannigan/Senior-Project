import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import rcParams
plt.style.use('seaborn-dark-palette')
rcParams['font.family'] = 'serif'
import numpy as np
import pandas as pd
import os

base_dir = '/home/luke/final_project/syncProbability'

trial_list = [25, 50, 75, 100, 125]
dur = 100
N = 4
var = [N, dur]
cols = ['y', 'b', 'c', 'm', 'g', 'r']

# probSync_4g3t50.csv ... 4 nodes with global coupling solved over 3 trials and 50 durations with new definition

header = 'Probability of synchronisation against coupling strength \
for\n{0} oscillators over {1} periods with varied number of trials'
fig = plt.figure(1)
ax = plt.subplot(111)
ax.set_title(header.format(*var))
ax.set_xlabel('Net Coupling J')
ax.set_ylabel('P(sync)')

for i, trial in enumerate(trial_list):
	data_name = 'syncProb_%dg%dt.csv' % (N, trial)
	frame = pd.read_csv(os.path.join(base_dir, data_name))
	data_arr = frame.to_numpy()

	j_list = np.array([row[0] for row in data_arr])
	j_list = j_list*6.
	prob_list = np.array([row[1] for row in data_arr])
	p_err = np.array([row[2] for row in data_arr])

	label = '%d trials' % trial
	
	ax.errorbar(j_list, prob_list, yerr=p_err, fmt='%s.-'%cols[i],  ecolor=cols[i], elinewidth=1.5, label=label)
	
#Jc = 3*(np.sqrt(8/np.pi))

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
ax.plot([Jc, Jc], [0, 1.1], color='black', ls='-', lw=1.5)
ax.text(Jc-0.04, -0.05, '$J_{c}$')

out_dir = '/home/luke/final_project/convergence'

''' convTrials_4g50_temp.eps | 4 node global network with 50 trials '''
fig_name ='convTrials_%dg%d_temp.png' % (N, dur)
fig.savefig(os.path.join(out_dir, fig_name))

plt.show()
	
	
