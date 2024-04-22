from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from model import PGN, GNN, symmetrize

is_pgn = input("is pgn (y/n): ") == "y"
train_data_path = input("train data: ")
# train_data_path = input("val data: ")
model_path = input("model: ")
save_path = input("save to: ")
n_epochs = int(input("epochs: "))
# patience = int(input("patience: "))

train_data = np.load(train_data_path)
E = torch.tensor(train_data["E"])
M_ground = torch.tensor(train_data["M"])
P_ground = torch.tensor(train_data["P"])
Y_ground = torch.tensor(train_data["Y"])
sym_P_ground = symmetrize(P_ground)

if is_pgn:
  model = PGN(query_size = 32, enc_size = 32, hidden_size = 32)
else:
  model = GNN(enc_size = 32, hidden_size = 32)
if model_path != "":
  model.load_state_dict(torch.load(model_path))

opt = optim.Adam(model.parameters(), lr = 0.005)

for epoch in tqdm(range(n_epochs)):
  opt.zero_grad()

  if is_pgn:
    Y, A, M = model(E, sym_P_ground)

    loss1 = F.binary_cross_entropy(Y, Y_ground)
    loss2 = F.cross_entropy(A, P_ground[1:]) / E.shape[0]
    loss3 = F.binary_cross_entropy(M, M_ground)
    loss = loss1 + loss2 + loss3

    print(f"{loss.item()} ({loss1.item()} + {loss2.item()} + {loss3.item()})")
  else:
    Y = model(E)

    loss = F.binary_cross_entropy(Y, Y_ground)

    print(loss.item())

  loss.backward()
  opt.step()

torch.save(model.state_dict(), save_path)
