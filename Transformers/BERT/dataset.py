import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset

class BertDataset(Dataset):
    def __init__(self,dataset,tokenizer,config):
        self.data = dataset
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = self.data[index]['text']
        if len(text.strip())==0:
            return self.__getitem__((index+1)%len(self.data))
        
        encoded = self.tokenizer(
            text,
            max_length = self.config.max_length,
            padding = "max_length",
            truncation = True,
            return_tensors = 'pt'
        )

        return encoded['input_ids'].squeeze(0)


def bert_mlm(tokens, tokenizer, mask_prob=0.8, replace_prob=0.1):
    batch_size, seq_len = tokens.shape
    device = tokens.device
    
    labels = tokens.clone()
    selected = torch.rand(batch_size, seq_len, device=device) < 0.15 
    augmented_tokens = tokens.clone()    
    rand_vals = torch.rand(batch_size, seq_len, device=device)
    
    mask_positions = selected & (rand_vals < mask_prob)
    augmented_tokens[mask_positions] = tokenizer.mask_token_id
    
    replace_positions = selected & (rand_vals >= mask_prob) & (rand_vals < mask_prob + replace_prob)
    num_replacements = int(replace_positions.sum().item())
    if num_replacements > 0:
        augmented_tokens[replace_positions] = torch.randint(0, tokenizer.vocab_size, (num_replacements,), device=device)
    labels[~selected] = -100
    
    return augmented_tokens, labels


