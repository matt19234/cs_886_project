import torch
import numpy as np
from sklearn.metrics import f1_score
from model import PGN, GNN

is_pgn = input("is pgn (y/n): ") == "y"
model_path = input("model: ")
test_data_path = input("test data: ")

if is_pgn:
  model = PGN(query_size = 32, enc_size = 32, hidden_size = 32)
else:
  model = GNN(enc_size = 32, hidden_size = 32)
model.load_state_dict(torch.load(model_path))

test_data = np.load(test_data_path)
E = torch.tensor(test_data["E"])
Y_ground = torch.tensor(test_data["Y"])
P_ground = torch.tensor(test_data["P"])

Y = model.predict(E)

scores = f1_score(Y_ground, Y, average = None)

print(f"batch 0 preds = {Y[:, 0]}")
print(f"batch 0 ground = {Y_ground[:, 0]}")
print(f"f1 scores = {scores}")
print(f"avg f1 score = {np.mean(scores)} +- {np.std(scores)}")
