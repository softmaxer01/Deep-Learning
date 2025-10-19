import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self,ds,tokenizer_src,tokenizer_tgt,src_lang,tgt_lang,seq_len):
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")],dtype = torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")],dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")],dtype = torch.int64)
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair["translation"][self.src_lang]
        tar_text = src_target_pair["translation"][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tar_text).ids

        enc_num_padding_tokens = self.seq_len-len(enc_input_tokens)-2
        dec_num_padding_tokens = self.seq_len-len(dec_input_tokens)-1

        if enc_num_padding_tokens<0 or dec_num_padding_tokens<0:
            raise ValueError
        
        encoder_inputs = torch.concat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens,dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token.item()]*enc_num_padding_tokens,dtype=torch.int64)
            ]
        )
        
        decoder_inputs = torch.concat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens,dtype = torch.int64),
                torch.tensor([self.pad_token.item()]*dec_num_padding_tokens,dtype=torch.int64)
            ]
        )


        label = torch.concat(
            [
                torch.tensor(dec_input_tokens,dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token.item()]*dec_num_padding_tokens,dtype=torch.int64)
            ]
        )

        assert encoder_inputs.size(0) == self.seq_len
        assert decoder_inputs.size(0) == self.seq_len
        assert label.size(0) == self.seq_len


        return {
            "encoder input": encoder_inputs,
            "decoder input": decoder_inputs,
            "encoder mask": (encoder_inputs != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder mask": (decoder_inputs != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_inputs.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tar_text
        }
    

def causal_mask(size):
    mask = torch.tril(torch.ones((1,size, size))).type(torch.int)
    return mask == 1 
