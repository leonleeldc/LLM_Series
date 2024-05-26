import numpy as np
import threading


def worker(A, B, C, row):
  n = len(B[0])
  for j in range(n):
    C[row][j] = sum(A[row][k] * B[k][j] for k in range(len(B)))


def multi_threaded_matrix_multiplication(A, B):
  m = len(A)
  n = len(B[0])
  C = np.zeros((m, n))

  threads = []
  for i in range(m):
    t = threading.Thread(target=worker, args=(A, B, C, i))
    threads.append(t)
    t.start()

  for t in threads:
    t.join()

  return C


# Example usage
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = multi_threaded_matrix_multiplication(A, B)
print(C)