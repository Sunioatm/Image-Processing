import numpy as np
# a = np.array([[1,2,3,3,2,1],
#              [4,5,6,6,5,4],
#              [7,8,9,9,8,7]])
# mask = np.array([[1/9,1/9,1/9],
#                  [1/9,1/9,1/9],
#                  [1/9,1/9,1/9]])
# test = a[0:3,0:3]
# print(np.sum(test*mask))

a = np.array([[1,2,3],
             [4,5,6],
             [7,8,9]])
mask = np.array([[1/9,1/9,1/9],
                 [1/9,1/9,1/9],
                 [1/9,1/9,1/9]])

result1 = a*mask
result2 = np.multiply(a,mask)
result3 = np.dot(a,mask)

print(result1)
print(result2)
print(result3)

# a_flat = a.flatten()
# mask_flat = mask.flatten()
# print(np.dot(a_flat,mask_flat))

# print(np.sum(np.dot(a,mask)))

