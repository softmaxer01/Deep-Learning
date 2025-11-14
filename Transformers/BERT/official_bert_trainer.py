import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from config import BertConfig, BertDatasetconfig
from dataset import BertDataset, bert_mlm
from transformers import AutoTokenizer, AutoModelForMaskedLM, get_linear_schedule_with_warmup
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import os
from tqdm import tqdm
import numpy as np

def main():
    dataset_config = BertDatasetconfig()
    model_config = BertConfig()
    
    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    print("Dataset loaded successfully")
    print(dataset)
    
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    
    train_data = BertDataset(dataset['train'], tokenizer, dataset_config)
    val_data = BertDataset(dataset['validation'], tokenizer, dataset_config)
    test_data = BertDataset(dataset['test'], tokenizer, dataset_config)
    
    train_loader = DataLoader(
        train_data, 
        batch_size=model_config.batch_size, 
        shuffle=True, 
        num_workers=2
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=model_config.batch_size, 
        num_workers=2
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=model_config.batch_size, 
        num_workers=2
    )
    
    device = torch.device(model_config.device)
    model = model.to(device)
    
    print(f"Model loaded on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=0.01,
        eps=1e-8
    )
    
    epochs = 3
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    def validate(model, val_loader, loss_fn):
        """Validation function to evaluate model on validation set"""
        model.eval()
        total_val_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for tokens in tqdm(val_loader, desc="Validating", leave=False):
                tokens = tokens.to(device)
                
                masked_tokens, labels = bert_mlm(tokens, tokenizer)
                
                outputs = model(input_ids=masked_tokens, labels=labels)
                loss = outputs.loss
                
                total_val_loss += loss.item()
                num_batches += 1
        
        avg_val_loss = total_val_loss / num_batches
        return avg_val_loss
    
    def train(model, train_loader, val_loader, optimizer, scheduler, epochs):
        """Training function"""
        train_losses_per_batch = []
        val_losses_per_epoch = []
        
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            epoch_train_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, tokens in enumerate(progress_bar):
                tokens = tokens.to(device)
                
                masked_tokens, labels = bert_mlm(tokens, tokenizer)
                
                optimizer.zero_grad()
                
                outputs = model(input_ids=masked_tokens, labels=labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                batch_loss = loss.item()
                train_losses_per_batch.append(batch_loss)
                epoch_train_loss += batch_loss
                global_step += 1
                
                current_lr = scheduler.get_last_lr()[0]
                
                # Update progress bar every step
                progress_bar.set_postfix({
                    'Loss': f'{batch_loss:.4f}',
                    'LR': f'{current_lr:.2e}'
                })
                
                # Print detailed info only every 200 steps to keep it clean
                if batch_idx % 200 == 0 and batch_idx > 0:
                    progress_bar.write(f'Step {global_step}: Loss={batch_loss:.4f}, LR={current_lr:.2e}')
            
            avg_epoch_train_loss = epoch_train_loss / len(train_loader)
            
            print("\nRunning validation...")
            avg_val_loss = validate(model, val_loader, criterion)
            val_losses_per_epoch.append(avg_val_loss)
            
            print(f'\n✓ Epoch {epoch+1}/{epochs} completed - Train Loss: {avg_epoch_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"New best validation loss: {best_val_loss:.4f}")
                
                save_dir = "./best_bert_model"
                os.makedirs(save_dir, exist_ok=True)
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                print(f"Model saved to {save_dir}")
        
        return train_losses_per_batch, val_losses_per_epoch
    
    print("Starting training...")
    train_losses_per_batch, val_losses_per_epoch = train(
        model, train_loader, val_loader, optimizer, scheduler, epochs
    )
    
    print("Evaluating on test set...")
    test_loss = validate(model, test_loader, criterion)
    print(f"Final test loss: {test_loss:.4f}")
    
    torch.save({
        'train_losses': train_losses_per_batch,
        'val_losses': val_losses_per_epoch,
        'test_loss': test_loss,
        'config': model_config
    }, 'training_history.pt')
    
    print("Training completed!")
    print(f"Training losses saved. Total batches: {len(train_losses_per_batch)}")
    print(f"Validation losses per epoch: {val_losses_per_epoch}")

if __name__ == "__main__":
    main()
