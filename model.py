import torch
import torch.nn as nn

def symmetrize(P):
  return torch.logical_or(P, P.mT)

class Proc(nn.Module):
  def __init__(self, enc_size, hidden_size):
    super(Proc, self).__init__()

    self.enc_size = enc_size
    self.hidden_size = hidden_size

    self.U = nn.Linear(enc_size * 2, hidden_size)
    self.M1 = nn.Linear(enc_size, enc_size)
    self.M2 = nn.Linear(enc_size, enc_size)

  def forward(self, z, P):
    # z : B x K x enc_size
    # P : B x K x K
    (B, K, Z) = z.shape

    m1z = self.M1(z)
    m2z = self.M2(z)

    m = torch.empty((B, K, self.enc_size))
    for b in range(B):
      for i in range(K):
        J = P[b, :, i].nonzero().squeeze(-1)
        m[b, i] = torch.max(torch.relu(m1z[b, i].unsqueeze(0) + m2z[b, J]), 0).values

    # m = m1z
    # for b in range(B):
    #   for i in range(K):
    #     J = P[b, :, i].nonzero().squeeze(-1)
    #     m[b, i] += torch.max(m2z[b, J], 0).values
    # m = torch.relu(m)

    return torch.relu(self.U(torch.cat((z, m), -1)))

class PGN(nn.Module):
  def __init__(self, query_size, enc_size, hidden_size):
    super(PGN, self).__init__()

    self.query_size = query_size
    self.enc_size = enc_size
    self.hidden_size = hidden_size

    self.enc = nn.Linear(2 + hidden_size, enc_size)
    self.proc = Proc(enc_size, hidden_size)
    self.dec = nn.Linear(hidden_size + enc_size, 1)

    self.mask = nn.Linear(enc_size + hidden_size, 1)
    self.Wq = nn.Linear(hidden_size, query_size)
    self.Wk = nn.Linear(hidden_size, query_size)

  def forward(self, E, P):
    (L, B, K, _) = E.shape

    h = torch.zeros(B, K, self.hidden_size)
    Y = torch.empty((L, B))
    A = torch.empty((L, B, K, K))
    M = torch.empty((L, B, K))

    for t in range(L):
      z = self.enc(torch.cat((E[t], h), -1))
      h = self.proc(z, P[t])
      # don't threshold for training
      Y[t] = torch.sigmoid(self.dec(torch.cat((torch.max(z, 1).values, torch.max(h, 1).values), -1))).squeeze(-1)

      q = self.Wq(h)
      k = self.Wk(h)
      A[t] = q @ k.mT # no softmax because cross-entropy loss does that already

      M[t] = torch.sigmoid(self.mask(torch.cat((z, h), -1))).squeeze(-1)

    return Y, A, M
    
  def predict(self, E):
    # l = length of input sequence
    # b = batch size
    # k = number of nodes
    (L, B, K, _) = E.shape

    h = torch.zeros(B, K, self.hidden_size)
    P = torch.empty((B, K, K))
    P[:] = torch.eye(K)
    Y = torch.empty((L, B))

    for t in range(L):
      z = self.enc(torch.cat((E[t], h), -1))
      h = self.proc(z, P)
      # threshold predictions
      Y[t] = torch.sigmoid(self.dec(torch.cat((torch.max(z, 1).values, torch.max(h, 1).values), -1))).squeeze(-1) > 0.5

      q = self.Wq(h)
      k = self.Wk(h)
      A = torch.softmax(q @ k.mT, -1) # B x K x K

      M = torch.sigmoid(self.mask(torch.cat((z, h), -1))) # B x K x 1
      mu = M > 0.5
      P = symmetrize(mu * P + (~mu) * (torch.arange(K) == A.max(-1).indices.unsqueeze(-1)))

    return Y

class GnnProc(nn.Module):
  def __init__(self, enc_size, hidden_size):
    super(GnnProc, self).__init__()

    self.enc_size = enc_size
    self.hidden_size = hidden_size

    self.U = nn.Linear(enc_size * 2, hidden_size)
    self.M1 = nn.Linear(enc_size, enc_size)
    self.M2 = nn.Linear(enc_size, enc_size)

  def forward(self, z):
    # z : B x K x enc_size
    (B, K, _) = z.shape

    m1z = self.M1(z)
    m2z = self.M2(z)

    m = torch.relu(m1z + torch.max(m2z, 1).values.unsqueeze(1))

    return torch.relu(self.U(torch.cat((z, m), -1)))

class GNN(nn.Module):
  def __init__(self, enc_size, hidden_size):
    super(GNN, self).__init__()

    self.enc_size = enc_size
    self.hidden_size = hidden_size

    self.enc = nn.Linear(2 + hidden_size, enc_size)
    self.proc = GnnProc(enc_size, hidden_size)
    self.dec = nn.Linear(hidden_size + enc_size, 1)

    # dummy params, just here so we can load model params from a PGN into a GNN
    # in order to speed up training
    self.mask = nn.Linear(enc_size + hidden_size, 1)
    self.Wq = nn.Linear(hidden_size, 32)
    self.Wk = nn.Linear(hidden_size, 32)
    
  def forward(self, E):
    # l = length of input sequence
    # b = batch size
    # k = number of nodes
    (L, B, K, _) = E.shape

    h = torch.zeros(B, K, self.hidden_size)
    Y = torch.empty((L, B))

    for t in range(L):
      z = self.enc(torch.cat((E[t], h), -1))
      h = self.proc(z)
      # threshold predictions
      Y[t] = torch.sigmoid(self.dec(torch.cat((torch.max(z, 1).values, torch.max(h, 1).values), -1))).squeeze(-1)

    return Y
    
  def predict(self, E):
    return (self.forward(E) > 0.5).to(torch.float32)
