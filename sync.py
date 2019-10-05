from Kuramoto import *
import matplotlib.pyplot as plt
plt.style.use('dark_background')

model = Kuramoto(0.)

z1 = np.array([])
J_max = 10.
J_min = .0

J_list = np.linspace(0., 10., 21)

fig1 = plt.figure(1)
fig1.suptitle('Complex order parameter integration for\n$\omega$ = [{0:.2f}, {1:.2f}], initial $\\theta$ = [{2:.2f}, {3:.2f}]'\
.format(model.w[0], model.w[1], model.theta0[0], model.theta0[1]))

fig2 = plt.figure(2)
fig2.suptitle('$\\theta$ integration over interval for\n$\omega$ = [{0:.2f}, {1:.2f}], initial $\\theta$ = [{2:.2f}, {3:.2f}]'\
.format(model.w[0], model.w[1], model.theta0[0], model.theta0[1]))
i = 1

for j in J_list:

	model.J = j*model.A
	times, thetas = model.solve(end=3e2)
	
	_z = model.order(thetas)
	z1 = np.append(z1, _z[-1])
	
	if (j % (J_max/4.)) == 0. and j < J_max:
		
		ax1 = fig1.add_subplot(2,2,i)
		ax1.set_title('J = '+ str(j))
		ax1.plot(_z.real, _z.imag, 'c-', label='J={0:.2f}'.format(j))
		ax1.plot([_z[-1].real, _z[-1].real], [_z[-1].imag, _z[-1].imag], 'yo')
		ax1.set_xlim(-1, 1)
		ax1.set_ylim(-1, 1)
		
		
		ax2 = fig2.add_subplot(2,2,i)
		ax2.set_title('J = '+ str(j))
		ax2.plot(times, thetas, 'm-')
		ax2.set_ylim(0, 1)
		
		
		i+=1
		
fig1.tight_layout(rect=[0, 0.03, 1, 0.9])
fig2.tight_layout(rect=[0, 0.03, 1, 0.9])

header = 'Order Parameter magnitude against coupling strength\nfor $\omega$ = [{2:.2f}, {3:.2f}] and initial $\\theta$ = [{4:.2f}, {5:.2f}]'
var = (J_min,J_max,model.w[0],model.w[1], model.theta0[0], model.theta0[1])

plt.figure(3)
plt.title(header.format(J_min,J_max,model.w[0],model.w[1], model.theta0[0], model.theta0[1]))
plt.ylabel('Order parameter magnitude')
plt.xlabel('Coupling constant')
plt.plot(J_list, abs(z1), 'ro-')
plt.ylim(0., 1.)

plt.show()


