# model constants
NUM_TOKENS = 128    # vocab size
BATCH_SIZE = 32
BLOCK_SIZE = 128    # or seq_len 
EMBEDDING_DIM = 96  # n_embd in nanoGPT
N_HEAD = 6          # n_head in nanoGPT
D_HEAD = EMBEDDING_DIM // N_HEAD  # 16
N_LAYER = 2         # n_layer in nanoGPT
FFN_MUL = 2
DROPOUT = 0.2

# training constants
BATCH_SCALE = 2  # increased from 1
BATCH_SIZE = 4 * BATCH_SCALE
NUM_BATCHES = int(5e4) // BATCH_SCALE  # reduced from 1e5
GRADIENT_ACCUMULATE_EVERY = 4 // BATCH_SCALE
LEARNING_RATE = 1e-4 * (BATCH_SCALE ** 0.5)
VALIDATE_EVERY = 100
GENERATE_EVERY = 500  # reduced from 1000
GENERATE_LENGTH = 512  # reduced from 1024
SEQ_LEN = 512  # reduced from 1024