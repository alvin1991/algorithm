import numpy as np
import matplotlib.pyplot as plt
import math
import random


class adrc_atp ( object ):

	r = 0
	h = 0
	h0 = 0
	N0 = 0
	fh = 0
	r = 0
	x1 = 0
	x2 = 0

	def __init__ (atp,fhan_fun,fsg_fun,sign_fun,val_r,var_h,var_N0):
		atp.r = val_r
		atp.h = var_h
		atp.N0 = var_N0
		atp.fhan = fhan_fun
		atp.fsg = fsg_fun
		atp.sign = sign_fun

def sign(val):
	
	if val>1e-6:
		val  = 1
	elif val < -1e-6:
		val = -1
	else:
		val = 0

	return val


# x1(k+1) = x1(k) + h * x2(k)
# x2(k+1) = x2(k) - r0 * u(k)
# fhan(x1,x2,r0,h0)

def fhan(atp,excpt):

	x1_delta = atp.x1 - excpt

	atp.h0 = atp.N0 * atp.h
	d = atp.r * atp.h0 * atp.h0
	a0 = atp.x2 * atp.h0
	y = x1_delta + a0

	a1 = math.sqrt(d*(d+8*abs(y)))
	a2 = a0 + sign(y)*(a1-d)/2
	a = (a0 + y)*fsg(y,d) + a2*(1 - fsg(y,d))

	atp.fh = -atp.r * (a/d) * fsg(a,d) - atp.r * sign(a)*(1-fsg(a,d))

	atp.x1 = atp.x1 + atp.h * atp.x2
	atp.x2 = atp.x2 + atp.h * atp.fh

	return

def fsg(x,d):

	output = (sign(x+d) - sign(x-d))/2

	return output

def setup():
	
	ob = adrc_atp(fhan,fsg,sign,300000,0.005,2)

	#set cycle times
	size=500

	#set X axis value
	X  = np.array(range(0,size))

	Y_h0         = np.array(range(size), dtype = float)
	Y_a0         = np.array(range(size), dtype = float)
	Y_s          = np.array(range(size), dtype = float)
	Y_fh         = np.array(range(size), dtype = float)
	Y_x1         = np.array(range(size), dtype = float)
	Y_x2         = np.array(range(size), dtype = float)


	for k in np.arange ( size ):

		Y_s[k]  = 1000*math.sin(k%180/57.29) + random.randint(-200,200)
	 	ob.fhan(ob,Y_s[k])
	 	Y_h0[k] = ob.h0
	 	Y_fh[k] = ob.fh
	 	Y_x1[k] = ob.x1
	 	Y_x2[k] = ob.x2

	        

	fig = plt.figure()
	ax1 = fig.add_subplot(3, 1, 1)
	ax2 = fig.add_subplot(3, 1, 2)
	ax3 = fig.add_subplot(3, 1, 3)

	ax1.set_xlabel("cycle time")
	ax1.set_ylabel("rate")
	ax1.plot(X, Y_h0, color='red', label='h0')
	ax1.legend()

	ax2.set_xlabel("cycle time")
	ax2.set_ylabel("rate")
	ax2.plot(X, Y_x1,  color='green', label='x1')
	ax2.plot(X, Y_s,  color='red', label='signal')
	ax2.legend()

	ax3.set_xlabel("cycle time")
	ax3.set_ylabel("rate")
	# ax3.plot(X, Y_fh,  color='blue', label='fh')
	ax3.plot(X, Y_x2,  color='blue', label='x2')	
	ax3.legend()
	

	plt.show()


if __name__ == "__main__":
    setup ()