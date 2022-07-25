from turtle import back
import numpy as np 

energy = np.array([[1, 4, 3, 5, 2],
                   [3, 2, 5, 2, 3],
                   [5, 2, 4, 2, 1]])

track = np.copy(energy)
backtrack = np.zeros_like(track)
for i in range(1, energy.shape[0]):
    for j in range(energy.shape[1]):
        if j == 0:
            idx = np.argmin(track[i-1, j:j+2])
            backtrack[i, j] = idx + j
            track[i, j] += track[i-1, j + idx]
        else:
            idx = np.argmin(track[i-1, j-1:j+2])
            backtrack[i, j] = idx + j-1
            track[i, j] += track[i-1, idx + j-1]

j = np.argmin(track[-1])
path = str(j)
for i in range(energy.shape[0]-1, 0, -1):
    j = backtrack[i, j]
    path = str(backtrack[i, j]) + '-->' + path
print('minimum sum energy:', min(track[-1]))
print('Path:', path.rstrip('-->'))

print("Track matrix: ")
for i in range(len(track)):
    for j in range(len(track[0])):
        print(str(track[i][j]) + " ", end = "")
    print("\n")

print("-----------------------")

print("Backtrack matrix: ")
for i in range(len(backtrack)):
    for j in range(len(backtrack[0])):
        print(str(backtrack[i][j]) + " ", end = "")
    print("\n")