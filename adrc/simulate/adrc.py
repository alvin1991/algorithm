import numpy as np
import matplotlib.pyplot as plt
import math
import random


class adrc_t ( object ):
	# atp
	r = 0
	h = 0
	h0 = 0
	N0 = 0
	fh = 0
	r = 0
	x1 = 0
	x2 = 0
	# eso
	e = 0
	z1 = 0
	z2 = 0
	z3 = 0
	beta01 = 100
	beta02 = 1000
	beta03 = 2000
	fe = 0
	fe1 = 0
	y = 0
	u = 0
	b0 = 0.001
	alpha1 = 0.8
	alpha2 = 1.5
	zeta = 50

	def __init__ (adrc,fhan_fun,fsg_fun,sign_fun,fal_fun,eso_fun,val_r,var_h,var_N0):
		# atp
		adrc.r = val_r
		adrc.h = var_h
		adrc.N0 = var_N0
		adrc.fhan = fhan_fun
		adrc.fsg = fsg_fun
		adrc.sign = sign_fun

		# eso
		adrc.fal = fal_fun
		adrc.eso = eso_fun

# 
# atp sign function
# 
def sign(val):
	
	if val>1e-6:
		val  = 1
	elif val < -1e-6:
		val = -1
	else:
		val = 0

	return val


# 
# atp fhan function
# 
# x1(k+1) = x1(k) + h * x2(k)
# x2(k+1) = x2(k) - r0 * u(k)
# fhan(x1,x2,r0,h0)

def fhan(adrc,excpt):

	x1_delta = adrc.x1 - excpt

	adrc.h0 = adrc.N0 * adrc.h
	d = adrc.r * adrc.h0 * adrc.h0
	a0 = adrc.x2 * adrc.h0
	y = x1_delta + a0

	a1 = math.sqrt(d*(d+8*abs(y)))
	a2 = a0 + sign(y)*(a1-d)/2
	a = (a0 + y)*fsg(y,d) + a2*(1 - fsg(y,d))

	adrc.fh = -adrc.r * (a/d) * fsg(a,d) - adrc.r * sign(a)*(1-fsg(a,d))

	adrc.x1 = adrc.x1 + adrc.h * adrc.x2
	adrc.x2 = adrc.x2 + adrc.h * adrc.fh

	return

# 
# atp fsg function
# 
def fsg(x,d):

	output = (sign(x+d) - sign(x-d))/2

	return output

# 
# eso fal function
# 
def fal(e,alpha,zeta):

	fal_output  = 0
	s = (sign(e + zeta) - sign( e - zeta))/2
	fal_output = e * s/(math.pow(zeta,1-alpha)) +  math.pow(abs(e),alpha) * sign(e) * (1 - s)

	return fal_output


# 
# eso eso function
# 
def eso(adrc):

	adrc.e = adrc.z1 - adrc.y
	adrc.fe = fal(adrc.e,0.5,adrc.h)
	adrc.fe1 = fal(adrc.e,0.25,adrc.h)

	adrc.z1 = adrc.z1 + adrc.h * ( adrc.z2 - adrc.beta01 * adrc.e )
	adrc.z2 = adrc.z2 + adrc.h * ( adrc.z3 - adrc.beta02 * adrc.fe + adrc.b0 * adrc.u)
	adrc.z3 = adrc.z3 + adrc.h * ( -adrc.beta03 * adrc.fe1 )

	return

# 
# setup function
# 
def setup():
	
	ob = adrc_t(fhan,fsg,sign,fal,eso,300000,0.005,2)

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
	Y_z1         = np.array(range(size), dtype = float)
	Y_z2         = np.array(range(size), dtype = float)
	Y_fe         = np.array(range(size), dtype = float)
	Y_fe1        = np.array(range(size), dtype = float)
	Y_e        	 = np.array(range(size), dtype = float)

	for k in np.arange ( size ):

		Y_s[k]  = 100*math.sin(k%180/57.29) + random.randint(-200,200)
		# atp
	 	ob.fhan(ob,Y_s[k])

	 	# eso
	 	ob.y = ob.x1
	 	ob.eso(ob)


	 	Y_h0[k] = ob.h0
	 	Y_fh[k] = ob.fh
	 	Y_x1[k] = ob.x1
	 	Y_x2[k] = ob.x2

	 	Y_z1[k] = ob.z1
	 	Y_z2[k] = ob.z2

		Y_e[k]  = ob.e
	 	Y_fe[k] = ob.fe
	 	Y_fe1[k] = ob.fe1

	        

	fig = plt.figure()
	ax1 = fig.add_subplot(3, 1, 1)
	ax2 = fig.add_subplot(3, 1, 2)
	ax3 = fig.add_subplot(3, 1, 3)

	ax1.set_xlabel("cycle time")
	ax1.set_ylabel("rate")
	# ax1.plot(X, Y_h0, color='red', label='h0')
	ax1.plot(X, Y_e, color='green', label='e')
	ax1.plot(X, Y_fe, color='red', label='fe')
	ax1.plot(X, Y_fe1, color='blue', label='fe1')
	ax1.legend()

	ax2.set_xlabel("cycle time")
	ax2.set_ylabel("rate")
	ax2.plot(X, Y_x1,  color='red', label='x1')
	ax2.plot(X, Y_s,  color='green', label='signal')
	ax2.plot(X, Y_z1,  color='blue', label='z1')
	ax2.legend()

	ax3.set_xlabel("cycle time")
	ax3.set_ylabel("rate")
	# ax3.plot(X, Y_fh,  color='blue', label='fh')
	ax3.plot(X, Y_x2,  color='red', label='x2')	
	ax3.plot(X, Y_z2,  color='blue', label='z2')
	ax3.legend()
	

	plt.show()


if __name__ == "__main__":
    setup ()