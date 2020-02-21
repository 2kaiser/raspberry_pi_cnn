import numpy as np

#Step 1: Generate a 2-dim all-zero array A, with the size of 9 x 6 (row x column).
A = np.zeros((9,6))
print("A is: ")
print(A)
#Step 2: Create a block-I shape by replacing certain elements from 0 to 1 in array A
A[0][1:5] = 1 #top of I
A[1][1:5] = 1 #top of I
A[2][2:4] = 1 #middle of I
A[3][2:4] = 1 #middle of I
A[4][2:4] = 1 #middle of I
A[5][2:4] = 1 #middle of I
A[6][2:4] = 1 #middle of I
A[7][1:5] = 1 #bottom of I
A[8][1:5] = 1 #bottom of I
print("I shaped A is: ")
print(A)
################################################################################################################################
#todo
#Step 3: Generate a 2-dim array B by filling zero-vector at the top and bottom of the array A.
#make a zero array of desired size and then enclose the previous array around it
B = np.zeros((11,6))
B[:A.shape[0],:A.shape[1]] = A
#add in a row of zeros and remove the last row
Z = np.concatenate(([[0,0,0,0,0,0]],B[0:10][:]))
print("B is: ")
print(Z)
################################################################################################################################
C =(np.arange(66).reshape(11, 6))+1
print("C is: ")
print(C)
################################################################################################################################
D = np.multiply(Z,C)
print("D is: ")
print(D)
################################################################################################################################
E = np.zeros(26)
index = 0
for i in range(D.shape[0]):
    for j in range(D.shape[1]):
        if(D[i][j] != 0):
            #print(D[i][j])
            E[index] = D[i][j]
            index = index + 1

print("E is: ")
print(E)
################################################################################################################################
max, min = E.max(), E.min()
index = 0
F = np.zeros((11,6))
for i in range(D.shape[0]):
    for j in range(D.shape[1]):
        if(D[i][j] != 0):
            F[i][j] = (D[i][j] - min) / (max - min)
            index = index + 1
print("F is: ")
print(F)
#Step 8: Find the element in F with the closest absolute value to 0.25 and print it on screen.
curr_closest = 10000
for i in range(D.shape[0]):
    for j in range(D.shape[1]):
        if(abs(curr_closest-.25) > abs(F[i][j]-.25)):
            curr_closest =F[i][j]
print("Closest value is: ")
print(curr_closest)
