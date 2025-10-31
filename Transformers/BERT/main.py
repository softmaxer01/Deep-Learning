import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from config import BertConfig, BertDatasetconfig
from dataset import BertDataset,bert_mlm
from model import Encoder
from transformers import AutoTokenizer
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


dataset_config = BertDatasetconfig()
model_config = BertConfig()

print("loading dataset")
dataset = load_dataset("wikitext","wikitext-2-raw-v1",)
print("dataset loaded successfully")
print(dataset)
print("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

train_data = BertDataset(dataset['train'],tokenizer,dataset_config)
val_data = BertDataset(dataset['validation'],tokenizer,dataset_config)
test_data = BertDataset(dataset['test'],tokenizer,dataset_config)

train_loader = DataLoader(train_data,batch_size=32,shuffle=True,num_workers=2)
val_loader = DataLoader(val_data,batch_size=32,num_workers=2)
test_loader = DataLoader(test_data,batch_size=32,num_workers=2)

model_config.vocab_size = tokenizer.vocab_size
model = Encoder(model_config)

lm_head = nn.Linear(model_config.d_model,model_config.vocab_size)

device = model_config.device
model = model.to(device)
lm_head = lm_head.to(device)


optimizer = optim.AdamW(
    list(model.parameters()) + list(lm_head.parameters()),
    lr=5e-5,
    weight_decay=0.01
)

epochs = 3
total_steps = len(train_loader) * epochs
warmup_steps = int(0.1 * total_steps) 
warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
criterion = nn.CrossEntropyLoss(ignore_index=-100)


def validate(model, lm_head, val_loader, loss_fn):
    """Validation function to evaluate model on validation set"""
    model.eval()
    lm_head.eval()
    
    total_val_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for tokens in val_loader:
            tokens = tokens.to(device)
            
            masked_tokens, labels = bert_mlm(tokens, tokenizer)
            
            enc_out = model(masked_tokens)
            logits = lm_head(enc_out)
            
            loss = loss_fn(logits.view(-1, model_config.vocab_size), labels.view(-1))
            total_val_loss += loss.item()
            num_batches += 1
    
    avg_val_loss = total_val_loss / num_batches
    return avg_val_loss


def train(model, lm_head, train_loader, val_loader, optimizer, scheduler, loss_fn, eps):
    # Lists to store losses for each batch
    train_losses_per_batch = []
    val_losses_per_epoch = []
    
    global_step = 0
    
    for ep in range(eps):
        model.train()
        lm_head.train()
        epoch_train_loss = 0
        
        for batch_idx, tokens in enumerate(train_loader):
            tokens = tokens.to(device)

            masked_tokens, labels = bert_mlm(tokens, tokenizer)
            optimizer.zero_grad()
            enc_out = model(masked_tokens)
            logits = lm_head(enc_out)

            loss = loss_fn(logits.view(-1, model_config.vocab_size), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(lm_head.parameters()), 1.0)
            optimizer.step()
            scheduler.step()  
            
            batch_loss = loss.item()
            train_losses_per_batch.append(batch_loss)
            epoch_train_loss += batch_loss
            global_step += 1
            
            if batch_idx % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f'Epoch {ep+1}, Step {batch_idx}, Global Step {global_step}, Train Loss: {batch_loss:.4f}, LR: {current_lr:.2e}')
        
        avg_epoch_train_loss = epoch_train_loss / len(train_loader)
        
        avg_val_loss = validate(model, lm_head, val_loader, loss_fn)
        val_losses_per_epoch.append(avg_val_loss)
        
        print(f'Epoch {ep+1} completed. Avg Train Loss: {avg_epoch_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        print('-' * 60)
    
    return train_losses_per_batch, val_losses_per_epoch


train_losses_per_batch, val_losses_per_epoch = train(model, lm_head, train_loader, val_loader, optimizer, scheduler, criterion, eps=3)
