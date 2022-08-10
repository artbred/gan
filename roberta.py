from transformers import RobertaTokenizer, RobertaModel, pipeline
import torch

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

def create_embedding(text):
    with torch.no_grad():
        text = text[:512]
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        return torch.squeeze(output.pooler_output, 0)