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
import numpy as np
from scipy.special import jn,jn_zeros	#Bessel functions

class HankelTransform:
	def __init__(self,Rmax,order=0,N=500):
		c = jn_zeros(order,N+1)

		self.r = c[:N] *Rmax/c[N]			#Radius vector
		self.v = c[:N] /(2*np.pi*Rmax)	#Frequency vector

		Jn,Jm = np.meshgrid(c[:N],c[:N])

		self._C = 2/c[N] *jn(order,Jn*Jm/c[N]) / (
				abs(jn(order+1,Jn)) * abs(jn(order+1,Jm)))

		self._m1 = abs(jn(order+1,c[:N]))/Rmax
		self._m2 = self._m1*Rmax/c[N] /(2*np.pi*Rmax)
		
	def fht(self,f):
		return np.dot(self._C,f/self._m1)*self._m2
						
	def iht(self,F):
		return np.dot(self._C,F/self._m2)*self._m1
