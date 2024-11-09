import torch
from transformers import AutoTokenizer
from architecture import Transformer
import os

EXPERIMENT_DIRECTORY = "runs/shakespeare-test"


device = "mps" if torch.backends.mps.is_available() else "cpu"




net = Transformer()
net.to(device)
net.load_state_dict(torch.load(os.path.join(EXPERIMENT_DIRECTORY, "ckpt", "latest.pt"), weights_only=True))

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

input_text = input("Prompt: ")

max_length = 100

input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

generated_text = input_text
for _ in range(max_length):
    with torch.no_grad():

        outputs = net(input_ids)
        #logits = outputs.logit()
        print(outputs.shape)
        next_token_id = torch.argmax(outputs[:, -1, :], dim=-1)  
        print(next_token_id)

        input_ids = torch.cat((input_ids, next_token_id.unsqueeze(-1)), dim=1)


        predicted_token = tokenizer.decode(next_token_id.item())
        generated_text += predicted_token
        print(predicted_token, end="")

print(generated_text)