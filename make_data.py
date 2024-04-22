import numpy as np

class DSU:
  def __init__(self, k):
    self.parent = np.arange(k)
    self.rank   = np.random.random(k)

  def find(self, u):
    if self.parent[u] != u:
      self.parent[u] = self.find(self.parent[u])

    return self.parent[u]

  def union_sets(self, u, v):
    u_root = self.find(u)
    v_root = self.find(v)

    if u_root == v_root:
      return False

    if self.rank[u_root] < self.rank[v_root]:
      self.parent[u_root] = v_root
    else:
      self.parent[v_root] = u_root

    return True

  def as_adj_mat(self):
    k = len(self.parent)
    A = np.zeros((k, k))
    A[np.arange(k), self.parent] = 1
    # A[self.parent, np.arange(k)] = 1
    return A


n = int(input("n: "))
k = int(input("k: "))
l = int(input("l: "))
f = input("save to: ")

E = np.empty((l, n, k, 2), dtype = np.float32)
M = np.empty((l, n, k), dtype = np.float32)
P = np.empty((l + 1, n, k, k), dtype = np.float32)
Y = np.empty((l, n), dtype = np.float32)

for b in range(n):
  dsu = DSU(k)
  P[0, b] = dsu.as_adj_mat()
  for t in range(l):
    u = np.random.randint(k)
    v = np.random.randint(k)
    Y[t, b] = dsu.union_sets(u, v)
    P[t + 1, b] = dsu.as_adj_mat()
    E[t, b, :, 0] = dsu.rank
    E[t, b, u, 1] = 1
    E[t, b, v, 1] = 1
    M[t, b] = np.all(P[t, b] == P[t + 1, b], -1)

# print(E)
# print(P)
# print(Y)
np.savez(f, E = E, M = M, P = P, Y = Y)
