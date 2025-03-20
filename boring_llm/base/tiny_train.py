import os
import git
from datetime import datetime
import wandb
import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from boring_llm.base.tiny import TinyTransformerWrapper, TinyDecoder, TinyAutoregressiveWrapper
from boring_utils.utils import cprint, tprint
from boring_utils.helpers import DEBUG
# import os; os.environ['DEBUG'] = '3'


# constants - adjusted for 4090
BATCH_SCALE = 2  # increased from 1
BATCH_SIZE = 4 * BATCH_SCALE
NUM_BATCHES = int(5e4) // BATCH_SCALE  # reduced from 1e5
GRADIENT_ACCUMULATE_EVERY = 4 // BATCH_SCALE
LEARNING_RATE = 1e-4 * (BATCH_SCALE ** 0.5)
VALIDATE_EVERY = 100
GENERATE_EVERY = 500  # reduced from 1000
GENERATE_LENGTH = 512  # reduced from 1024
SEQ_LEN = 512  # reduced from 1024
BEST_VAL_LOSS = float('inf')
RUN_NAME = f"enwik8-boring-transformer_{datetime.now().strftime('%Y%m%d_%H%M')}"
REPO_ROOT = git.Repo(search_parent_directories=True).working_tree_dir
DATA_DIR = os.path.join(REPO_ROOT, 'data')
MODEL_SAVE_DIR = os.path.join(REPO_ROOT, 'checkpoints')

assert GENERATE_LENGTH <= SEQ_LEN, f"GENERATE_LENGTH ({GENERATE_LENGTH}) must be less than or equal to SEQ_LEN ({SEQ_LEN})"


# helpers
def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))


# instantiate model with TinyTransformerWrapper and TinyDecoder
model = TinyTransformerWrapper(
    num_tokens=256,
    max_seq_len=SEQ_LEN,
    dim=512,
    n_layers=6,
    n_head=8,
    d_head=64,  # 512 // 8
    ffn_mul=4,
    dropout=0.1,
    transform_layer=TinyDecoder
)

model = TinyAutoregressiveWrapper(model)
model.cuda()

# prepare enwik8 data
with gzip.open(os.path.join(DATA_DIR, 'enwik8.gz')) as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    train_x, valid_x = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(train_x), torch.from_numpy(valid_x)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True))

# optimizer
wandb.init(
    project="boring-transformer", 
    name=RUN_NAME, 
    config={
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "gradient_accumulation": GRADIENT_ACCUMULATE_EVERY,
    }
)
wandb.watch(model)

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training
os.makedirs(f"{MODEL_SAVE_DIR}/{RUN_NAME}", exist_ok=True)

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(next(train_loader))
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    print(f'training loss: {loss.item()}')
    wandb.log({"train_loss": loss.item()}, step=i)

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = model(next(val_loader))
            print(f'validation loss: {loss.item()}')
            wandb.log({"val_loss": loss.item()}, step=i)
            
            if loss < BEST_VAL_LOSS:
                BEST_VAL_LOSS = loss
                torch.save(model.state_dict(), f"{MODEL_SAVE_DIR}/{RUN_NAME}/best_model.pth")

    if i % GENERATE_EVERY == 0:
        tprint('Generating...')
        model.eval()
        inp = random.choice(val_dataset)[:-1]
        prime = decode_tokens(inp)
        cprint(prime, new_line=True)

        sample = model.generate(
            start_tokens=inp.unsqueeze(0),  # Modified to match your API
            seq_len=GENERATE_LENGTH,
            temperature=0.8  # Added temperature parameter
        )

        output_str = decode_tokens(sample[0])  # Get first batch item
        cprint(output_str, new_line=True)

        torch.save(model.state_dict(), f"{MODEL_SAVE_DIR}/{RUN_NAME}/model_{i}.pth")