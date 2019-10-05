import numpy as np
from scipy.integrate import odeint

class Kuramoto:

	def __init__(self, J = None, N = 2, homo = True):
		'''set initial values & attributes'''
		self.N = N
		self.A = np.array([[0.,1.],[1.,0.]])
		
		w = np.random.normal(0, .1, self.N) # mu = 0, sigma = 1
		theta0 = np.random.uniform(0, 2*np.pi, self.N)	
		
		if homo and J == None:
			J = np.random.uniform(0, 1)*self.A
		elif homo and (type(J) == float or type(J) == int):
			J = J*self.A
		else:
			J = np.random.uniform(0, 1, size=(self.N,self.N))
			np.fill_diagonal(J, 0)								  
		
		self.w = w
		self.J = J
		self.theta0 = theta0
		self.thetas = np.zeros_like(theta0)
		
	def __call__(self, _theta, t = 0):
		'''Kuramoto model ODE'''
		_theta = self.theta0
		theta_dot = np.zeros_like(_theta)
		
		for i in range(0, self.N): # i --> row, links --> column
			links = np.where(self.J[i] != 0.)
			theta_dot[i] = self.w[i] + \
			np.sum( self.J[i][links]*np.sin(_theta[links] - _theta[i]) )
		
		return theta_dot
		
	def solve(self, end = 100):
		'''integrates Kuramoto ODE'''
		t = np.linspace(0, end, end+1)
		self.thetas = odeint(self, self.theta0, t)
		
		return t, self.thetas
		
	def reset(self, choose_w = True, choose_theta = True):
		'''resets phase & frequency to new random values'''
		if choose_w:
			self.w = np.random.normal(0, 1, self.N)
		if choose_theta:
			self.theta0 = np.random.uniform(0, 2*np.pi, self.N)
		
	def order(self, _thetas):
		'''calulates order parameters for thetas'''
		z_list = np.array([])
		
		for _theta in _thetas:
			z = (1/self.N)*np.sum(np.exp(_theta*1j))
			z_list = np.append(z_list, z)
		
		return z_list
	
	def sample(self, trials):
		'''averages over set number of trials'''
		for _ in trials:
			times, thetas = self.solve()
			self.reset()

		

	
