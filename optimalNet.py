import time
start = time.time()
from Kuramoto import *
import pandas as pd
import os

run = 10

def run_time(start):
	s = time.time() - start
	h = s // (60.**2)
	m = s // 60.
	s = s % 60.
	return h, m, s

netJ = 3.
geometry = 'global'
model = Kuramoto(J=netJ, homo=False)


model.J = np.array([[0., 0., 1.5, 0.],
      			    [0., 0., 0., 1.5],
					[1.5, 0., 0., 0.],
					[0., 1.5, 0., 0.]])

numBonds = int( (model.N**2 - model.N) / 2 )
margin = 5e-4
trials = 100
dur = 50
xmax = 1.

var = [geometry, model.N, trials, dur, netJ, run]
bonds = [' j%d'%(n+1) for n in range(0,numBonds)]

# optimalNet_4g3.csv | 4 nodes with global coupling, net coupling of 3
data_name = 'optimalNet_{1}{0[0]}{4:.0f}_cross{5}.csv'.format(*var)
base_dir = '/home/luke/final_project/optimalNetwork'

def period(omega_arr):
	''' period of the smallest omega gap  '''
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
		#theta = model.solve(end=T, last=True)
		
		theta_dot = model(theta)
		phi_dot = abs( np.amax(theta_dot) - np.amin(theta_dot) )
		
		if phi_dot <= margin:
			syncs += 1
			
	prob = syncs/trials	
	err = np.sqrt((prob*(1 - prob))/trials)	
	
	return prob, err

################
# Append data file
frame = pd.read_csv(os.path.join(base_dir, data_name))
frame = frame.query('Accept == True').iloc[-1]  
data_arr = frame.to_numpy()

P = frame[1]
err = frame[2]

J = np.copy(model.J)
ind = 4
for row in range(0, numBonds):
	for step in range(0, model.N - row - 1):
		col = row + step + 1
		J[row][col] = frame[ind] 
		J[col][row] = frame[ind] 
		ind += 1
################
#print(J)
#exit()

'''################
# New initial coupling
J = np.copy(model.J)
print('\n\tInitial J\n', np.round(J, 3))

P, err = test(J)
print('\n\tInitial P: {:.2f} %'.format(100.*P))
print('\tTimestamp: %.0f.%.0f.%.0f'%(run_time(start)))


# New data file
data = {'Accept':True, 'Prob':P, 'error':err, 'Random float':0.}

ind = 0
for row in range(0, numBonds):
	for step in range(0, model.N - row - 1):
		col = row + step + 1
		key = bonds[ind]
		data[key] = J[row][col]
		ind += 1

data_frame = pd.DataFrame(data, index=[0])
data_frame.to_csv(os.path.join(base_dir, data_name), index=False, encoding='utf-8', float_format='%.4f')
'''################
#print(J)
#exit()

stagnantIters = 0
while stagnantIters < 1000:
	newJ = np.copy(J)
	
	print('\n\tCurrent J\n', np.round(J, 4))
			
	# randint: upper value is exclusive !
	row1 = np.random.randint(0, model.N - 1)
	step1 = np.random.randint(0, model.N - 1 - row1)
	col1 = row1 + step1 + 1

	row2 = np.random.randint(0, model.N - 1)
	step2 = np.random.randint(0, model.N - 1 - row2)
	col2 = row2 + step2 + 1

	while row1 == row2 and col1 == col2:
		row2 = np.random.randint(0, model.N - 1)
		step2 = np.random.randint(0, model.N - 1 - row2)
		col2 = row2 + step2 + 1

	x = np.random.uniform(0.0001, xmax) # incl low, excl high
	j1 = newJ[row1, col1]
	j2 = newJ[row2, col2]
	
	if (j1 == 0.) and (j2 == 0.):
		print('\n\t\tSkip')
		continue
	print('\tIterations without improvement: %d'%stagnantIters)

	newJ[row1, col1] = j1*(1. - x)
	newJ[row2, col2] = j2 + j1*x
	#reflect about diagonal
	newJ[col1, row1] = j1*(1. - x)
	newJ[col2, row2] = j2 + j1*x
	
	newP, err = test(newJ)
	
	indices = [x, row1+1, col1+1, row2+1, col2+1]
	print('\n\tRedistribute {:.3f} ({}, {}) --> ({},{})\n'.format(*indices), np.round(newJ, 4), '\n\n')
	print('\tProposed P(sync): {:.2f} %'.format(100.*newP))
	print('\tCurrent P(sync): {:.2f} %\n'.format(100.*P))
	
	if newP > P:
		J = np.copy(newJ)
		P = newP
		print('\n\tAccept')
		accept=True
		xmax = 1.
		stagnantIters = 0
	
	else:
		print('\n\tReject')
		accept=False	
		stagnantIters += 1
	
	data = {'Accept':accept, 'Probability':newP, 'Error':err, 'Random float':x}
	# add coupling data
	ind = 0
	for row in range(0, numBonds):
		for step in range(0, model.N - row - 1):
			col = row + step + 1
			key = bonds[ind]
			data[key] = newJ[row][col]
			ind += 1
	
	frame = pd.DataFrame(data, index=[0])
	frame.to_csv(os.path.join(base_dir, data_name), index=False, header=False, mode='a', encoding='utf-8', float_format='%.4f')
	
print('\n\t Best J with %:.2f P(sync)\n'%max_P, np.round(max_J, 4))
print('\n\tRuntime: %.0f.%.0f.%.0f'%(run_time(start)),'\n')


