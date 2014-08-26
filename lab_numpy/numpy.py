# coding: utf-8
x = np.array([0, 1, 2, 3])
y = np.array([-1, 0.2, 0.9, 2.1])
A = np.vstack([x, np.ones(len(x))]).T
A
m, c = np.linalg.lstsq(A, y)[0]
import matplotlib.pyplot as plt
plt.plot(x, y, 'o', label='Original data', markersize=10)
plt.plot(x, m*x + c, 'r', label='Fitted line')
plt.legend()
file = open('data.txt')
import numpy as np
data = np.genfromtxt(file, delimiter=',')
data
file.close()
M = data[:,:2]
M
np.hstack(M, np.full(20),1)
np.hstack(M, np.full(20,1),1)
np.hstack((M, np.full(20),1))
np.hstack((M, np.full((20,1),1)))
M
b = data[:,2:]
b
a = np.linalg.lstsq(M,b)
a
M * a
data
np.vstack(([1,2,1,0,0,0],[0,0,0,1,2,1]))
data[0]
m1 = data[0,:2]
m1
m
m1
m1[:-1]
m1[::-1]
data[0,:2]
data[0,:2].appennd(1)
data[0,:2].append(1)
a = data[0,:2]
np.append(a,1)
np.append(data[0,:2],1)
np.pad(np.append(data[0,:2],1),3,'constant',0)
np.pad(np.append(data[0,:2],1),3,'constant', constant_values=0)
np.pad(np.zeros(1,6), 3, 'constant', np.append(data[0,:2],1))
data
np.pad(np.zeros(1,6), 3, 'constant', constant_values=np.append(data[0,:2],1))
np.pad(np.zeros(1,6), 3, 'constant',4)
np.pad(np.zeros(1,6), 3, 'constant', constant_values=7)
np.pad(np.zeros(1,6), (3,), 'constant', constant_values=7)
np.zeros(1,6)
np.pad(np.zeros((1,6)), 3, 'constant', np.append(data[0,:2],1))
np.pad(np.zeros((1,6)), 3, 'constant', constant_values=np.append(data[0,:2],1))
np.pad(np.zeros((1,6)), 3, 'constant', (np.append(data[0,:2],1)))
np.pad(np.zeros((1,6)), 3, 'constant', constant_values = (np.append(data[0,:2],1)))
np.pad(np.zeros((1,6)), 3, 'constant', constant_values = tuple(np.append(data[0,:2],1)))
np.pad(np.zeros((1,6)), 3, 'constant', constant_values=(1,2,1))
np.pad(np.zeros((1,6)), 3, 'constant', constant_values=(1,2))
np.pad(np.zeros((1,6)), (3,), 'constant', constant_values=(1,2))
np.pad(np.zeros((1,6)), (1,), 'constant', constant_values=(1,2))
np.pad(np.zeros((1,6)), (0,), 'constant', constant_values=(1,2))
np.pad(np.zeros((1,6)), 3, 'constant', constant_values=1)
np.pad(np.zeros((1,6)), (3,0), 'constant', constant_values=1)
np.pad(np.zeros((1,6)), (0,3), 'constant', constant_values=1)
np.pad(np.zeros((1,6)), (0,3), 'constant', constant_values=1)
np.pad(np.zeros((1,6)), (3,0), 'constant', constant_values=1)
np.pad(np.zeros((1,6)), (3,0,0), 'constant', constant_values=1)
np.pad(np.zeros((1,6)), (1,0,0), 'constant', constant_values=1)
np.pad(np.zeros((1,6)), (1,0), 'constant', constant_values=1)
np.pad(np.zeros((1,6)), (0.1), 'constant', constant_values=1)
np.pad(np.zeros((1,6)), (0,0), 'constant', constant_values=1)
np.pad(np.zeros((1,6)), (0,1), 'constant', constant_values=1)
np.pad(np.zeros((1,6)), (1,0), 'constant', constant_values=1)
np.pad(np.zeros((1,6)), (3,0), 'constant', constant_values=1)
np.pad(np.zeros((1,6)), (3,3), 'constant', constant_values=1)
np.pad(np.zeros((1,6)), (2,2), 'constant', constant_values=1)
np.pad(np.zeros((1,6)), (0,2), 'constant', constant_values=1)
np.pad(np.zeros(6), (0,2), 'constant', constant_values=1)
np.pad(np.zeros(6), (2,2), 'constant', constant_values=1)
np.pad(np.zeros(6), 3, 'constant', constant_values=data[0,:2])
np.pad(np.zeros(6), 3, 'constant', constant_values=tuple(data[0,:2]))
np.pad(np.zeros(6), 3, 'constant', constant_values=(6,7,1))
np.pad(np.zeros(6), 3, 'constant', constant_values=(6,7))
np.pad(np.zeros(6), 3, 'constant', constant_values=6)
np.pad(np.zeros(6), (3,0), 'constant', constant_values=6)
np.pad(data[0,:2], (0,3), 'constant', constant_values=0)
np.pad(np.append(data[0,:2],1), (0,3), 'constant', constant_values=0)
np.pad(np.append(data[0,:2],1), (3,0), 'constant', constant_values=0)
for row in data:
    np.hstack((    np.pad(np.append(row[:2],1), (0,3), 'constant', constant_values=0),    np.pad(np.append(row[:2],1), (3,0), 'constant', constant_values=0)))
    
for row in data:
    print(
    np.hstack((\
    np.pad(np.append(row[:2],1), (0,3), 'constant', constant_values=0),\
    np.pad(np.append(row[:2],1), (3,0), 'constant', constant_values=0))))
    
for row in data:
    print(
    np.hstack((\
    np.pad(np.append(row[:2],1), (0,3), 'constant', constant_values=0),\
    np.pad(np.append(row[:2],1), (3,0), 'constant', constant_values=0))))
    
for row in data:
    np.hstack((    np.pad(np.append(row[:2],1), (0,3), 'constant', constant_values=0),    np.pad(np.append(row[:2],1), (3,0), 'constant', constant_values=0)))
    
M = np.array()
M = np.empty()
M = np.empty(40,4)
M = np.empty(0)
m
M
for row in data:
    np.hstack((M,    np.hstack((    np.pad(np.append(row[:2],1), (0,3), 'constant', constant_values=0),    np.pad(np.append(row[:2],1), (3,0), 'constant', constant_values=0)))))
    
M
M = np.empty(0,6)
M = np.empty((0,6))
M
for row in data:
    np.hstack((M,    np.hstack((    np.pad(np.append(row[:2],1), (0,3), 'constant', constant_values=0),    np.pad(np.append(row[:2],1), (3,0), 'constant', constant_values=0)))))
    
np.pad(np.append(row[:2],1), (0,3), 'constant', constant_values=0)
M = np.zeros(0,6)
M = np.zeros((0,6))
M
for row in data:
    np.hstack((M,    np.hstack((    np.pad(np.append(row[:2],1), (0,3), 'constant', constant_values=0),    np.pad(np.append(row[:2],1), (3,0), 'constant', constant_values=0)))))
    
M = np.zeros((1,6))
M
for row in data:
    np.hstack((M,    np.hstack((    np.pad(np.append(row[:2],1), (0,3), 'constant', constant_values=0),    np.pad(np.append(row[:2],1), (3,0), 'constant', constant_values=0)))))
    
np.pad(np.append(row[:2],1), (0,3), 'constant', constant_values=0)
np.pad(np.append(row[:2],1), (3,0), 'constant', constant_values=0)
np.hstack((    np.pad(np.append(row[:2],1), (0,3), 'constant', constant_values=0),    np.pad(np.append(row[:2],1), (3,0), 'constant', constant_values=0)))
np.hstack((    np.pad(np.append(row[:2],1), (0,3), 'constant', constant_values=0),    np.pad(np.append(row[:2],1), (3,0), 'constant', constant_values=0))).reshape(2,6)
def create2rows(row):
    return np.hstack((    np.pad(np.append(row[:2],1), (0,3), 'constant', constant_values=0),    np.pad(np.append(row[:2],1), (3,0), 'constant', constant_values=0))).reshape(2,6)
M = np.zeros(0,6)
M = np.zeros((0,6))
for row in data:
    np.hstack((M, create2rows(row)))
    
M = np.zeros((2,6))
M
M = np.empty((2,6))
M
M = np.zeroes((2,6))
M = np.zeros((2,6))
for row in data:
    np.hstack((M, create2rows(row)))
    
M
for row in data:
    M = np.hstack((M, create2rows(row)))
    
M
M.shape
M.reshape(-1,4)
M.reshape(-1,6)
create2rows(data[0])
np.hstack(create2rows(data[0]), create2rows(data[1]))
np.hstack((create2rows(data[0]), create2rows(data[1])))
create2rows

def create2rows(row):
    return np.hstack((    np.pad(np.append(row[:2],1), (0,3), 'constant', constant_values=0),    np.pad(np.append(row[:2],1), (3,0), 'constant', constant_values=0))).ravel()
np.hstack((create2rows(data[0]), create2rows(data[1])))
np.hstack((create2rows(data[0]), create2rows(data[1]))).reshape(4,-1)
M = create2rows(data[0])
for row in data[1:]:
    np.hstack(M,create2rows(row))
    
for row in data[1:]:
    np.hstack((M,create2rows(row)))
    
M.reshape(40,-1)
M.shape
for row in data[1:]:
    M=np.hstack((M,create2rows(row)))
    
M.reshape(-1,6)

b = data[2:].reshape((-1,1))
b
b = data[0,2:]
b
for row in data:
    b=np.append(b, row[2:])
    
b.reshape((-1,1))
import numpy.linalg as la
a,e,r,s = la.lstsq(M,b)
b.reshape((-1,2))
a,e,r,s = la.lstsq(M,b)
a.shape
M.shape
b.shape
b
b = b.reshape(-1,1)
b
M = M.reshape((-1,6))
la.lstsq(M,b)
b.shape[0]
b
b.shape,M.shape
b
b.shape
b = b[2:]
b
b.shape
la.lstsq(M,b)
a,e,r,s = la.lstsq(M,b)
M * a
a
M.shape
M * a
d
a * M
M * a
M
a
M .* a
np.dot(M,a)
la.norm(np.dot(M, a-b))
a-b
a.shape
b.shape
la.norm(np.dot(M, a) - b)
r
e
e,r,s
e
r
e
la.norm(np.subtract(np.dot(M,a),b))
np.square(la.norm(np.subtract(np.dot(M,a),b)))

