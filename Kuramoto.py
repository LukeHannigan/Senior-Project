import numpy as np
from scipy.integrate import odeint

class Kuramoto:

	def __init__(self, J = None, N = 4, homo = True):
		'''set initial values & attributes'''
		self.N = N
		#self.A = np.array([[0.,1.],[1.,0.]])
		self.A = np.ones((self.N, self.N))
		for j in range(self.N):
			for i in range(self.N):
				if i == j:
					self.A[i][j] = 0.

		
		w = np.random.normal(0, 1, self.N) # mu = 0, sigma = 1
		self.w = w
		
		theta0 = np.random.uniform(0, 2*np.pi, self.N)
		self.theta0 = theta0	
		
		if homo and (J is None):
			numBonds = (self.N**2 - self.N)/2
			J = np.random.uniform(0.0001, 1.)*self.A
			J = J/(.5*np.sum(J))
		elif homo and J is not None:
			numBonds = (self.N**2 - self.N)/2
			J = (J/numBonds)*self.A
			
		elif not homo:
			J1 = np.copy(self.A)

			for row in range(0, self.N - 1):
				for step in range(0, self.N - row - 1):
					col = row + step + 1
					J1[row][col] = np.random.uniform(0.0001, 1.)
					J1[col][row] = J1[row][col]
					
			J1 = J1/(.5*np.sum(J1))
			
			if J is not None:
				J = J*J1 # *2. as np.sum
			else:
				J = J1
				
		self.J = J
		
		
		self.thetas = np.zeros_like(self.theta0)
		self.theta_dots = np.array([])
		
		self.phis = np.array([])
		self.phi_dots = np.array([])
		self.choose_w = True
		self.choose_theta = True
		self.t = 0
		
	def __call__(self, _theta = None, t = 0):
		'''Kuramoto model ODE'''
		if _theta is None:
			_theta = self.theta0
		theta_dot = np.zeros_like(_theta)
		
		for n in range(self.N): # n --> row, links --> column
			links = np.where(self.J[n] != 0.)
			theta_dot[n] = self.w[n] + \
			np.sum( self.J[n][links]*np.sin(_theta[links] - _theta[n]) )
			
		self.theta_dots = np.append(self.theta_dots, theta_dot)
		
		return theta_dot
		
	def solve(self, end = 3e2, last = False):
		'''integrates Kuramoto ODE'''
		times = np.linspace(0, end, end+1)
		self.thetas = odeint(self, self.theta0, times)
		self.t = int(end + 1.)
		
		#self.theta_dots = np.reshape(self.theta_dots, (self.t, self.N))
		if last == False:
			return times, self.thetas
		
		if last == True:
			return self.thetas[-1]
		
	def reset(self):
		'''resets phase & frequency to new random values'''
		if self.choose_w:
			self.w = np.random.normal(5, 1, self.N)
		if self.choose_theta:
			self.theta0 = np.random.uniform(0, 2*np.pi, self.N)
		
		self.thetas = np.array([])#np.zeros_like(self.theta0)
		self.theta_dots = np.array([])
		
		
	def order(self, _thetas = None):
		'''calulates order parameters for thetas'''
		if _thetas is None:
			_thetas = self.thetas
		
		z_list = np.array([])
		
		if _thetas.ndim == 1:
			_z = (1/self.N)*np.sum(np.exp(_thetas*1j))
			z_list = np.append(z_list, _z)
		
		else:
			for t in range(self.t):
				_z = (1/self.N)*np.sum(np.exp(_thetas[:][t]*1j))
				z_list = np.append(z_list, _z)
		
		return z_list
		
	def sampleOrder(self, trials, end = 3e2):
		'''averages order parameter in time for set # trials'''
		R_list = np.array([])
		
		for trial in range(trials):
			self.reset() # new phases
			times, thetas = self.solve(end)
			
			R = abs( self.order(thetas[-1]) )
			R_list = np.append(R_list, R)

		#R_list = np.reshape(R_list, (times.shape[0], trials))
		R_mean = np.mean(R_list)#, axis=1)

		return R_mean
				
		
#theta = np.array([row[n] for row in thetas])	

	
