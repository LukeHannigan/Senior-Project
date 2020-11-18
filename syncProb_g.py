import time
start = time.time()
from Kuramoto import *
import pandas as pd
import os

def run_time(start):
	s = time.time() - start
	h = s // (60.**2)
	m = s // 60. # floor division
	s = s % 60.
	return h, m, s

base_dir = '/home/luke/final_project/syncProbability'

geometry = 'global'
model = Kuramoto(J = 0.)

margin = 5e-4
dur = 100

def period(omega_arr):
	''' period of the slowest nodest '''
	w, N = sorted(omega_arr), len(omega_arr)
	wDiffs = [w[i + 1] - w[i] for i in range(N-1)]
	deltaW = min(wDiffs)
	if deltaW < 0.01:
		deltaW = 0.01
	return 2*np.pi/deltaW

def test(J_arr):	
	syncs = 0
	model.J = np.copy(J_arr)
	for trial in range(trials):
		print('\n\tTrial %d of %d'%(trial+1, trials))
		print('\tTimestamp: %.0f.%.0f.%.0f'%(run_time(start)))
		model.reset()
		
		theta = model.solve(end=dur*period(model.w), last=True)
		
		theta_dot = model(theta)
		phi_dot = abs( np.amax(theta_dot) - np.amin(theta_dot) )
		
		if phi_dot <= margin:
			syncs += 1
			
	prob = syncs/trials	
	err = np.sqrt((prob*(1 - prob))/trials)	
	
	return prob, err


trials_list = [5, 10, 25, 50, 75, 100, 125]
trials_list = trials_list[4:]
j_pts = 21
j_max = 1.
j_list = np.linspace(0, j_max, j_pts)

for trials in trials_list:
	print('\n\n\t\tTrials:', trials)
	var = [geometry, model.N, trials, dur]

	# probSync_4g3t50.csv ... 4 nodes with global coupling solved over 3 trials and 50 durations with new definition
	data_name = 'syncProb_{1}{0[0]}{2}t{3}.csv'.format(*var)
	data = {'Probability' : [], 'Error' : [], 'j' : []}
	frame = pd.DataFrame(data)
	frame.to_csv(os.path.join(base_dir, data_name), index=False)

	for j in j_list:
		_start = time.time()
		print('\n\t\tCoupling: %.2f of %.0f' % (j, j_max))
		print('\t\tTimestamp: %.0f.%.0f.%.0f'%(run_time(start)))
		
		model.J = np.copy(j*model.A)

		prob, err = test(model.J)	

		
		data = {'Probability' : prob, 'Error' : err, 'j' : j}
		frame = pd.DataFrame(data, index=[0])
		frame.to_csv(os.path.join(base_dir, data_name), index=False, header=False, mode='a', encoding='utf-8', float_format='%.3f')
		

print('\nRuntime: %.0f.%.0f.%.0f'%(run_time(start)),'\n')


