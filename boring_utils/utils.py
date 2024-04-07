import os
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import pprint
import inspect

# color patterns
RED_PATTERN = '\033[31m%s\033[0m'
GREEN_PATTERN = '\033[32m%s\033[0m'
BLUE_PATTERN = '\033[34m%s\033[0m'
PEP_PATTERN = '\033[36m%s\033[0m'
BROWN_PATTERN = '\033[33m%s\033[0m'
YELLOW_PATTERN = '\033[93m%s\033[0m'


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # apple silicon GPU
    else:
        device = torch.device("cpu")
    return device


def check_exists(data_folder, dataset_name):
    return os.path.exists(os.path.join(data_folder, dataset_name))


def mkdir(folder_path):
    Path(folder_path).mkdir(parents=True, exist_ok=False)


def init_graph(figsize=(10, 10), dpi=100):
    plt.style.use(["dark_background", "bmh"])
    plt.rc("axes", facecolor="k")
    plt.rc("figure", facecolor="k")
    plt.rc("figure", figsize=figsize, dpi=dpi)
    plt.rc("font", size=15)


def plot_data(X, y, figsize=(16, 16), save_fig=False, fig_path="temp.png"):
    """plot data generated from make_data.py"""
    plt.figure(figsize=figsize)
    plt.title("Dataset")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis")

    if save_fig:
        plt.savefig(fig_path)

    plt.show()


def plot_loss(
    train_losses,
    val_losses,
    num_epochs,
    figsize=(20, 10),
    save_fig=False,
    fig_path="temp.png",
):
    fig = plt.figure(figsize=figsize)
    plt.plot(np.arange(1, num_epochs + 1), train_losses, label="Train loss")
    plt.plot(np.arange(1, num_epochs + 1), val_losses, label="Validation loss")
    plt.xlabel("Loss")
    plt.ylabel("Epochs")
    plt.legend(loc="upper right")

    if save_fig:
        plt.savefig(fig_path)

    plt.show()


def deriv(func, input_, delta=0.001):
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)


def get_batch_np(data, block_size=1024, batch_size=32, device='cpu'):
    # Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive)
    # The shape of the tensor is defined by the variable argument size
    # 0 ~ len(data) - block_size with output shape of (batch_size,)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss_np(model, eval_iters, train_data, val_data, block_size=1024, batch_size=32, device='cpu'):
    out = {}
    model.eval()
    data_dic = {'train': train_data, 'val': val_data}
    for split, data in data_dic.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_np(data, block_size=block_size, batch_size=batch_size, device=device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def mprint(obj, magic_methods=False, private_methods=True, public_methods=True):
    # Split items based on their types
    
    if private_methods:
        magic_methods = [x for x in dir(obj) if x.startswith("__") and x.endswith("__")]
        private_methods = [x for x in dir(obj) if x.startswith("_") and not x in magic_methods]
        print("\n\033[93mPrivate Methods:\033[0m")
        for item in sorted(private_methods):
            print(f"    {item}")

        if magic_methods:
            print("\n\033[93mMagic Methods:\033[0m")
            for item in sorted(magic_methods):
                print(f"    {item}")
    
    if public_methods:
        public_methods = [x for x in dir(obj) if not x.startswith("_")]
        print("\n\033[93mPublic Methods:\033[0m")
        for item in sorted(public_methods):
            print(f"    {item}")


def get_layers(func):
    '''
    Usage:
      from boring_nn import pe
      pe_layers = get_layers(pe)
      pe_name = 'SinusoidalPositionalEncoding'
      pos_encoding = pe_layers[pe_name](d_model, dropout, max_len)
    '''
    func_layers = {}
    for name, obj in inspect.getmembers(func):
        if inspect.isclass(obj) and issubclass(obj, torch.nn.Module):
            # func_layers[name.lower()] = obj
            func_layers[name] = obj
    return func_layers


def cprint(*exprs, c=None, class_name=True):
    """
    Custom print function that prints the name of the variable/expression
    alongside its value.
    
    Parameters:
    - *exprs: Variable-length argument list of expressions to evaluate.
    - globals, locals (dict, optional): Optional dictionaries to specify the namespace 
      for the evaluation. This allows the function to access variables outside of its 
      local scope.
    - class_name (bool, optional): If True, prints the class name or function name along with the variable name.
    
    Example:
    x = 10
    y = 20
    cprint(x)
    # Output: x: 10
    
    cprint(x, y)
    # Output:
    # x: 10
    # y: 20
    
    cprint()
    # Output: (Empty line)
    """
    # Fetch the line of code that called cprint
    frame = inspect.currentframe().f_back
    call_line = inspect.getframeinfo(frame).code_context[0].strip()
    
    # Extract the arguments from the line
    arg_str = call_line[call_line.index('(') + 1:-1].strip()
    
    # Split the arguments by comma, keeping expressions intact
    arg_list = []
    bracket_count = 0
    current_arg = []
    for char in arg_str:
        if char == ',' and bracket_count == 0:
            arg_list.append(''.join(current_arg).strip())
            current_arg = []
        else:
            if char in '([{':
                bracket_count += 1
            elif char in ')]}':
                bracket_count -= 1
            current_arg.append(char)
    if current_arg:
        arg_list.append(''.join(current_arg).strip())
    
    # Check if there are any arguments
    if not arg_list or (len(arg_list) == 1 and not arg_list[0]):
        print()  # Print an empty line
        return
    
    for arg, expr in zip(arg_list, exprs):
        try:
            if class_name:
                # Get the class name or function name from the caller's frame
                class_or_func_name = frame.f_code.co_name
                if 'self' in frame.f_locals:
                    class_or_func_name = frame.f_locals['self'].__class__.__name__
                arg = f"{class_or_func_name} -> {arg}"
            
            if not c:
                print(YELLOW_PATTERN % f"{arg}:")
                pprint.pprint(expr)
            if c == 'red':
                print(RED_PATTERN % f"{arg}:")
                pprint.pprint(expr)
            elif c == 'green':
                print(GREEN_PATTERN % f"{arg}:")
                pprint.pprint(expr)
            elif c == 'blue':
                print(BLUE_PATTERN % f"{arg}:")
                pprint.pprint(expr)
            elif c == 'pep':
                print(PEP_PATTERN % f"{arg}:")
                pprint.pprint(expr)
            elif c == 'brown':
                print(BROWN_PATTERN % f"{arg}:")
                pprint.pprint(expr)
            elif c == 'normal':
                pprint.pprint(arg)

        except Exception as e:
            print(f"Error evaluating {arg}: {e}")


def sprint(*exprs, globals=None, locals=None):
    """
    Custom print function that prints the name of the variable/expression
    alongside its value.
    
    Parameters:
    - *exprs (str): Variable-length argument list of expressions to evaluate as strings.
    - globals, locals (dict, optional): Optional dictionaries to specify the namespace
      for the evaluation. This allows the function to access variables outside of its
      local scope.
    
    Example:
    x = 10
    y = 20
    sprint("x")
    # Output: x: 10
    
    sprint("x", "y")
    # Output:
    # x: 10
    # y: 20
    
    sprint()
    # Output: (Empty line)
    """
    # Check if there are any arguments
    if not exprs:
        print()  # Print an empty line
        return
    
    for expr in exprs:
        try:
            # Evaluate the expression
            value = eval(expr, globals, locals)
            print(f"\033[93m{expr}\033[0m: \n{value}\n")
        except Exception as e:
            print(f"Error evaluating {expr}: {e}")


# import torch 

# def count_unique(tensor):
#     # Calculate unique values and their counts
#     unique_values, counts = torch.unique(tensor, return_counts=True)

#     # Convert unique_values to a Python list
#     unique_values = unique_values.tolist()

#     # Convert counts to a Python list
#     counts = counts.tolist()

#     # Print the unique values and their counts
#     for value, count in zip(unique_values, counts):
#         print(f"Value: {value}, Count: {count}")
# 
#     print()
