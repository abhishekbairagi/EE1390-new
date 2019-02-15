import numpy as np 
import matplotlib.pyplot as plt 

A1 = np.array([[np.sqrt(2)],[-1]])
A1 = A1.T
K = A1.T/(4*np.sqrt(2))
print 'K' ,K
print 'A1' ,'A1'
print np.dot(K,A1)
dvec = np.array([-1,1])
omat = np.array([[0,1],[-1,0]])
def norm_vec(AB):
	return np.matmul(omat, np.matmul(AB,dvec))
def line_intersect(AD,CF):
	n1 = norm_vec(AD)
	n2 = norm_vec(CF)
	N = np.vstack((n1,n2))
	p=np.zeros(2)
	p[0] = np.matmul(n1, AD[:,0])
	p[1] = np.matmul(n2 , CF[: ,0])
	return np.matmul(np.linalg.inv(N),p)

for k in range(-40,40):
	if k!=0:
		A = np.array([-4*k,0])
		B = np.array([0,4*np.sqrt(2)*k])
		C = np.array([4.0/k,0])
		D = np.array([0,4*np.sqrt(2)/k])

		len = 100
		lam_1 = np.linspace(-100,100,len)

		x_AB = np.zeros((2,len))
		x_CD = np.zeros((2,len))

		for i in range(len):
			temp1 = A + lam_1[i]*(B-A)
			x_AB[:,i] = temp1.T 
			temp2 = D + lam_1[i]*(C-D)
			x_CD[:,i] = temp2.T 
		AB = np.vstack((A,B)).T
		CD = np.vstack((C,D)).T
		X=line_intersect(AB,CD)
		
		plt.plot(X[0],X[1],'o',color = "black")
		plt.plot(x_AB[0,:],x_AB[1,:], color = "yellow")
		plt.plot(x_CD[0,:],x_CD[1,:], color = "red")

r = np.array([4,4*np.sqrt(2)])

theta = np.linspace(0,(np.pi)/2,10000)
x1 = (r[0]*np.tan(theta))
x2 = (r[1]*(1/np.cos(theta)))  
plt.plot(x1 , x2,color = "blue")

theta = np.linspace(0,-(np.pi)/2,10000)
x1 = (r[0]*np.tan(theta))
x2 = (r[1]*(1/np.cos(theta)))  
plt.plot(x1 , x2,color = "blue")

theta = np.linspace(np.pi,3*(np.pi)/2,10000)
x1 = (r[0]*np.tan(theta))
x2 = (r[1]*(1/np.cos(theta)))  
plt.plot(x1 , x2,color = "blue")

theta = np.linspace(-np.pi,-3*(np.pi)/2,10000)
x1 = (r[0]*np.tan(theta))
x2 = (r[1]*(1/np.cos(theta)))  
plt.plot(x1 , x2,color = "blue")

plt.xlabel('$x$')
plt.xlabel('$y$')
plt.xlim(-25,25)
plt.ylim(-50,50)
#plt.axis("equal")
plt.grid()

plt.show()