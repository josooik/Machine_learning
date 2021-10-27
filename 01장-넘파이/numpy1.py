import numpy as np

array1 = np.array([1, 2, 3])
print(type(array1))
print(array1.shape)

array2 = np.array([[1, 2, 3], [2, 3, 4]])
print(type(array2))
print(array2.shape)

array3 = np.array([[1, 2, 3]])
print(type(array3))
print(array3.shape)

print(array1.ndim, array2.ndim, array3.ndim)

list1 = [1, 2, 3]
print(type(list1))
array1 = np.array(list1)
print(type(array1))

list2 = [1, 2, 'test']
array2 = np.array(list2)
print(type(array2))
print(array2, array2.dtype)

list3 = [1, 2, 3.0]
array3 = np.array(list3)
print(array3, array3.dtype)

array_int = np.array([1, 2, 3])
array_float = array_int.astype('float64')
print(array_float, array_float.dtype)

array_int1 = array_float.astype('int32')
print(array_int1, array_int1.dtype)

array_float1 = np.array([1.1, 2.1, 3.1])
array_int2 = array_float1.astype('int32')
print(array_int2, array_int2.dtype)

sequence_array = np.arange(10)
print(sequence_array)
print(sequence_array.dtype, sequence_array.shape)

sequence_array2 = np.arange(5, 10)
print(sequence_array2)
print(sequence_array2.dtype, sequence_array2.shape)

zeros_array = np.zeros((3, 2), dtype='int32')
print(zeros_array)
print(zeros_array.dtype, zeros_array.shape)

one_array = np.ones((3, 2))
print(one_array)
print(one_array.dtype, one_array.shape)

array1 = np.arange(10)
print("\n", array1)
array2 = array1.reshape(2, 5)
print("\n", array2)
array3 = array1.reshape(5, 2)
print("\n", array3)

array1 = np.arange(10)
print("\n", array1)
array2 = array1.reshape(-1, 5)
print("\n", array2)
array3 = array1.reshape(5, -1)
print("\n", array3)

array1 = np.arange(8)
print("\n", array1)
array3d = array1.reshape((2, 2, 2))
print("\n", array3d)

array5 = array3d.reshape(-1, 1)
print("\n", array5, array5.shape)

array6 = array1.reshape(1, -1)
print("\n", array6, array6.shape)

array1 = np.arange(1, 10)
print("\n", array1)
value = array1[2]
print("\n", value, type(value))

print(array1[-1], array1[-2])

array1[0] = 9
array1[8] = 0
print("\n", array1)

array1d = np.arange(1, 10)
print("\n", array1d)
array2d = array1d.reshape(3, 3)
print("\n", array2d)
print("\n", array2d[0, 0])
print("\n", array2d[0, 1])
print("\n", array2d[1, 0])
print("\n", array2d[2, 2])

array1 = np.arange(1, 10)
print("\n", array1)
array3 = array1[0:3]
print("\n", array3)
print("\n", type(array3))

array1 = np.arange(1, 10)
print("\n", array1)
array4 = array1[:3]
print("\n", array4)
array5 = array1[3:]
print("\n", array5)
array6 = array1[:]
print("\n", array6)

print("\n------------------------------------")

array1d = np.arange(1, 10)
print("\n", array1d)
array2d = array1d.reshape(3, 3)
print("\n", array2d)
print("\n", array2d[0:2, 0:2])
print("\n", array2d[1:3, 0:3])
print("\n", array2d[1:3, :])
print("\n", array2d[:, :])
print("\n", array2d[:2, 1:])
print("\n", array2d[:2, 0])

array1d = np.arange(1, 10)
print("\n", array1)
array2d = array1d.reshape(3, 3)
print("\n", array2d)
print("\n", array2d[0])
print("\n", array2d[1])
print("\n", array2d.shape)
print("\n", array2d[0].shape)
print("\n", array2d[1].shape)

array1d = np.arange(1, 10)
print("\n", array1d)
array2d = array1d.reshape(3, 3)
print("\n", array2d)
array3 = array2d[[0, 1], 2]
print("\n", array3)
array4 = array2d[[0, 1], 0:2]
print("\n", array4)
array5 = array2d[[0, 1]]
print("\n", array5)
array6 = array2d[[0, 2], 1]
print("\n", array6)

array1d = np.arange(1, 10)
print("\n", array1d)
array3 = array1d[array1d > 5]
print("\n", array3)

print("\n", array1d > 5)

org_array = np.array([3, 1, 9, 5])
print("\n", org_array)
sort_array1 = np.sort(org_array)
print("\n", sort_array1)
print("\n", org_array)
sort2_array = org_array.sort()
print("\n", sort2_array)
print("\n", org_array)

org_array = np.array([3, 1, 9, 5])
print("\n", org_array)
sort_array1 = np.sort(org_array)[::-1]
print("\n", sort_array1)
print("\n", org_array)

array2d = np.array([[8, 12], [7, 1]])
print("\n", array2d)
array0 = np.sort(array2d, axis=0)
print("\n", array0)
array1 = np.sort(array2d, axis=1)
print("\n", array1)

org_array = np.array([3, 1, 9, 5])
print("\n", org_array)
sort_indices = np.argsort(org_array)
print("\n", sort_indices)

org_array = np.array([3, 1, 9, 5])
print("\n", org_array)
sort_indices = np.argsort(org_array)[::-1]
print("\n", sort_indices)

name_array = np.array(['John', 'Mkie', 'Sarah', 'Kate', 'Samuel'])
score_array = np.array([78, 95, 84, 98, 88])
print("\n", score_array)
sort_socre = np.sort(score_array)
print("\n", sort_socre)
sort_indices = np.argsort(score_array)
print("\n", sort_indices)
print("\n", name_array[sort_indices])

A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8], [9, 10], [11, 12]])
dot_product = np.dot(A, B)
print("\n", A)
print("\n", B)
print("\n", dot_product)

A = np.array([[1, 2], [3, 4]])
transpose_mat = np.transpose(A)
print("\n", A)
print("\n", transpose_mat)