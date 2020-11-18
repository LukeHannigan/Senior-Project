import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.style.use('seaborn-dark-palette')
rcParams['font.family'] = 'serif'
import numpy as np
import pandas as pd
import os

run = 5

base_dir = '/home/luke/final_project/optimalNetwork'

data_name = 'optimalNet_4g3_cross%d.csv' % run
fig_name = 'optimalNet_4g3_cross%d.png' % run
frame = pd.read_csv(os.path.join(base_dir, data_name))
frame = frame.query('Accept == True')  
data_arr = frame.to_numpy()

p_list = np.array([row[1] for row in data_arr])
p_err = np.array([row[2] for row in data_arr])
iter_list = np.array(frame.index.tolist())

print('run=%d' % run)
print(iter_list[-1])

header='Probability of sync optimizing over number of iterations'

fig = plt.figure(1)
ax = plt.subplot(111)
ax.set_title(header)
ax.set_xlabel('Iterations')
ax.set_ylabel('P(sync)')
ax.errorbar(iter_list, p_list, yerr=p_err, fmt='r.-',  ecolor='r', elinewidth=1.5)
ax.set_ylim(0, 1)
ax.set_xlim(0)
ax.grid(which='both', ls='-', lw=1.)

fig.savefig(os.path.join(base_dir, fig_name))

plt.show()
