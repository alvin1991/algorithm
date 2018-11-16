import numpy as np
import matplotlib.pyplot as plt
import math
import random


class adrc_eso ( object ):

	r = 0
	h = 0
	h0 = 0
	N0 = 0
	fh = 0
	r = 0
	x1 = 0
	x2 = 0

	def __init__ (eso,fal_fun,sign_fun):
		eso.fal = fal_fun
		eso.sign = sign_fun

def sign(val):
	
	if val>1e-6:
		val  = 1
	elif val < -1e-6:
		val = -1
	else:
		val = 0

	return val



def fal(eso,e,alpha,zeta):

	fal_output  = 0
	s = (sign(e+zeta) - sign(e-zeta))/2
	fal_output = e * s/(math.pow(zeta,1-alpha)) +  math.pow(abs(e),alpha) * sign(e) * (1 - s)

	return fal_output


def fsig(x):


	fsig_output =  -math.pow(x,3) - x - 0.2 * 1 + 0.5 * sign(math.cos(x%360/57.29))

	return fsig_output


def setup():
	
	ob = adrc_eso(fal,sign)

	#set cycle times
	size=10

	#set X axis value
	X  = np.array(range(0,size))

	Y_fo   = np.array(range(size), dtype = float)
	Y_fsig   = np.array(range(size), dtype = float)

	for k in np.arange ( size ):

	 	
		Y_fo[k]  = ob.fal(ob,k,0.8,50)

		Y_fsig[k] = fsig(k)


	fig = plt.figure()
	ax1 = fig.add_subplot(2, 1, 1)
	ax2 = fig.add_subplot(2, 1, 2)

	ax1.set_xlabel("cycle time")
	ax1.set_ylabel("rate")
	ax1.plot(X, Y_fo, color='red', label='fo')
	ax1.legend()

	ax2.set_xlabel("cycle time")
	ax2.set_ylabel("rate")
	ax2.plot(X, Y_fsig,  color='green', label='signal')
	ax2.legend()

	# fig = plt.figure()
	# ax1 = fig.add_subplot(3, 1, 1)
	# ax2 = fig.add_subplot(3, 1, 2)
	# ax3 = fig.add_subplot(3, 1, 3)

	# ax1.set_xlabel("cycle time")
	# ax1.set_ylabel("rate")
	# ax1.plot(X, Y_h0, color='red', label='h0')
	# ax1.legend()

	# ax2.set_xlabel("cycle time")
	# ax2.set_ylabel("rate")
	# ax2.plot(X, Y_x1,  color='green', label='x1')
	# ax2.plot(X, Y_s,  color='red', label='signal')
	# ax2.legend()

	# ax3.set_xlabel("cycle time")
	# ax3.set_ylabel("rate")
	# # ax3.plot(X, Y_fh,  color='blue', label='fh')
	# ax3.plot(X, Y_x2,  color='blue', label='x2')	
	# ax3.legend()
	

	plt.show()


if __name__ == "__main__":
    setup ()