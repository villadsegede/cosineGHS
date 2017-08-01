#Port of matlab code by Manuel Guizar Sicairos based on
#
#  M. Guizar-Sicairos and J. C. Gutierrez-Vega, Computation of quasi-discrete
#  Hankel transforms of integer order for propagating optical wave fields,
#  J. Opt. Soc. Am. A 21, 53-58 (2004).
#
#Written by Villads Egede Johansen, Mar 2014
# Current e-mail: vej22 (a) cam.ac.uk
# Long term e-mail: villads (a) egede.com

from __future__ import division
import pylab as pl
from scipy.io import loadmat
from scipy.special import jn		#Bessel functions

cmat = loadmat("c.mat")["c"]

class HankelTransform():
	def __init__(self,Rmax,order=0,N=500):
		self._Rmax = Rmax
		self._order = order
		self._N = N

		c = cmat[self._order,:self._N+1]

		self.r = c[:N] *Rmax/c[N]			#Radius vector
		self.v = c[:N] /(2*pl.pi*Rmax)	#Frequency vector
		self._V = c[N] /(2*pl.pi*Rmax)	#Max frequency

		Jn,Jm = pl.meshgrid(c[:N],c[:N])

		self._C = 2/c[N] *jn(order,Jn*Jm/c[N]) / (
				abs(jn(order+1,Jn)) * abs(jn(order+1,Jm)))

		self._m1 = abs(jn(order+1,c[:N]))/Rmax
		self._m2 = self._m1*Rmax/self._V
		
	def fht(self,f):
		#Check if input is a function, and evaluate if so
		if hasattr(f,'__call__'):
			f = f(self.r)
		return pl.dot(self._C,f/self._m1) *self._m2
						
	def iht(self,F):
		#Check if input is a function, and evaluate if so
		if hasattr(F,'__call__'):
			f = f(self.r)
		return pl.dot(self._C,F/self._m2)*self._m1


if __name__ == "__main__":
	print("Running test of Hankel transform")
	print("(feel free to change the order from 0 to 4)")
	Rmax = 4.
	N = 250
	order = 0

	#Setup Hankel transformation
	ht = HankelTransform(Rmax,order,N)
	r = ht.r
	v = ht.v

	#Input function
	f = (r**order) * (r<=1)

	# The input can also be stated as a function:
	#  (to test this, uncomment these two lines and
	#   the plot function further below, and comment
	#   out the old plot command instead)
	#def f(x):
	#	return x**order *(x<=1)


	#Analytical solution to f
	vmax = 5.
	van = pl.linspace(1e-10,vmax,200)
	Fanalytic = jn(order+1,2*pl.pi*van)/van


	#Perform Hankel transform (and back)
	F = ht.fht(f)
	fback = ht.iht(F)

	pl.subplot("211")
	pl.plot(r,f)
	#pl.plot(r,f(r))
	pl.plot(r,fback,'.')
	pl.ylim(-0.1,1.1)
	pl.legend(["Input function","Retrieved function with IHT"])


	pl.subplot("212")
	pl.plot(v,F,'.')
	pl.plot(van,Fanalytic)
	pl.xlim(0,vmax)
	pl.legend(["Transformation result","Analytical solution"])
	pl.show()
