import sys
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

useSecondLevelPretraining = sys.argv[1]
input_text = sys.argv[2]


if(useSecondLevelPretraining):
  tok = AutoTokenizer.from_pretrained("./train_data_4k/checkpoint-1500")
  model = AutoModelForMaskedLM.from_pretrained("./train_data_4k/checkpoint-1500")
else:
  tok = AutoTokenizer.from_pretrained("bert-large-uncased")
  model = AutoModelForMaskedLM.from_pretrained("bert-large-uncased")

x = tok.encode(input_text)
idx = x.index(103)
x = torch.Tensor([x]).long()

l = model(x)[0]
l = l[0][idx].detach().cpu().numpy()
la = l.argsort()

for i in range(1, 11):
    print(tok.decode([la[-i]]))
