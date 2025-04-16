import tiktoken
import torch
from torch.utils.data import Dataset,DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, tokenizer:tiktoken.Encoding, raw_text:str, context_size:int, stride:int, ):
        self.input_ids = []
        self.target_ids = []
        self.tokenizer =  tokenizer

        token_ids = self.tokenizer.encode(raw_text)

        for i in range(0, len(token_ids) - context_size, stride):
            input = token_ids[i:i+context_size]
            target = token_ids[i+1:i+context_size+1]
            self.input_ids.append(torch.tensor(input))
            self.target_ids.append(torch.tensor(target))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return self.input_ids[i], self.target_ids[i]

def create_dataloader_v1(
    text: str,
    tokenizer:tiktoken.Encoding,
    batch_size: int = 4,
    context_size:int = 256,
    stride:int = 128,
    shuffle:bool=True,
    drop_last:bool=True,
    num_workers:int=0
):
    dataset = GPTDatasetV1(tokenizer, text, context_size, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader