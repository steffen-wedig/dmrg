import jax.numpy as jnp
import numpy as np 

a = 5
b= 4
c=3
d=3
s=6
n=3
m=2
o= 4
r=2


A = np.random.normal(size=(a,b,d,s))
B = np.random.normal(size=(b,c,m))
C = np.random.normal(size=(d,n,m,o))
D = np.random.normal(size=(n,r,a))

F = np.einsum("abds,bcm,dnmo,nra->cors",A,B,C,D)
print(F.shape)