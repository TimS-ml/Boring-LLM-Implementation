import os
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
import torch


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


def plot_model_boundary(
    X,
    y,
    model,
    data_type="swirl",
    torch_model=False,
    figsize=(16, 16),
    save_fig=False,
    fig_path="temp.png",
):
    """plot decision boundary of data generated from make_data.py"""
    plt.figure(figsize=figsize)

    # load pre-definied meshgrid range
    if data_type == "moon":
        xx, yy = np.meshgrid(np.arange(-1.5, 2.5, 0.01), np.arange(-1.5, 1.5, 0.01))
    else:
        # 'swirl' etc. centered
        mesh = np.arange(-1.1, 1.1, 0.01)
        xx, yy = np.meshgrid(mesh, mesh)

    if torch_model:
        model.to("cpu")
        with torch.no_grad():
            data = torch.from_numpy(
                np.vstack((xx.reshape(-1), yy.reshape(-1))).T
            ).float()
            Z = model(data).detach()
        Z = np.argmax(Z, axis=1).reshape(xx.shape)
    else:  # sklearn model
        data = np.vstack((xx.reshape(-1), yy.reshape(-1))).T
        Z = model.predict(data)
        Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.3)

    plt.title("Dataset and Decision Boundary")
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


def plot_func_and_deriv(
    func,
    fig_name=None,
    approxim=False,
    figsize=(10, 10),
    save_fig=False,
    fig_path="temp.png",
):
    plt.figure(figsize=figsize)

    fig, ax = plt.subplots()
    inputRange = np.arange(-5, 5, 0.01)
    ax.plot(inputRange, func.fn(inputRange), linewidth=3, label="Original Function")
    if callable(func.grad) and func.grad(inputRange):  # sometime func.grad is not implemented
        ax.plot(
            inputRange, func.grad(inputRange), linewidth=3, label="Derivate Function"
        )
    if approxim:
        ax.plot(
            inputRange,
            [deriv(func, i) for i in inputRange],
            linestyle="dashed",
            linewidth=3,
            label="Derivate Function Approximation",
        )
    if not fig_name:
        ax.set_title(f"Function: {func.name}")
    else:
        ax.set_title(fig_name)
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    ax.legend(loc="upper left")

    if save_fig:
        fig.savefig(fig_path)


def chain_forward(chain, inputRange):
    print(f"The length of function chain = {len(chain)}.")

    for func in chain:
        inputRange = func.fn(inputRange)

    return inputRange


def chain_deriv(chain, inputRange):
    chain_deriv = np.ones_like(inputRange)
    # chain_deriv = np.expand_dims(chain_deriv, axis=0)

    for func in chain:
        chain_deriv = chain_deriv * deriv(func, inputRange)
        inputRange = func.fn(inputRange)

    return chain_deriv.squeeze()


def plot_func_chain_and_deriv(
    func_chain, fig_name=None, figsize=(10, 10), save_fig=False, fig_path="temp.png"
):
    plt.figure(figsize=figsize)

    fig, ax = plt.subplots()
    inputRange = np.arange(-5, 5, 0.01)
    ax.plot(
        inputRange,
        chain_forward(func_chain, inputRange),
        linewidth=3,
        label="Original Function",
    )
    ax.plot(
        inputRange,
        chain_deriv(func_chain, inputRange),
        linestyle="dashed",
        linewidth=3,
        label="Derivate Function Approximation",
    )
    if not fig_name:
        func_name = " -> ".join([func.name for func in func_chain])
        ax.set_title(f"Function: {func_name}")
    else:
        ax.set_title(fig_name)
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    ax.legend(loc="upper left")

    if save_fig:
        fig.savefig(fig_path)


# [The spelled-out intro to neural networks and backpropagation: building micrograd - YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0)
def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.children:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


# def computation_graph(root):
#     # assert isinstance(root, Tensor) == True  # perform type check

#     node_style = {
#         "fontname": "Helvetica",
#         "fontsize": "12",
#         "fontcolor": "white",
#         "shape": "box",
#         "style": "filled",
#         "fillcolor": "#343434",
#     }
#     edge_style = {
#         "fontname": "Helvetica",
#         "fontsize": "10",
#         "fontcolor": "white",
#         "arrowhead": "open",
#         "color": "white",
#     }

#     nodes, edges = trace(root)
#     dot = Digraph(
#         format="png",
#         graph_attr={"rankdir": "TB", "bgcolor": "#222222"},
#         node_attr=node_style,
#         edge_attr=edge_style,
#     )

#     for n in nodes:
#         dot.node(
#             name=str(id(n)),
#             label="{ %s | data %.2f | grad %.2f }" % (n.label, n.data, n.grad),
#             shape="record",
#         )
#         if n.operation:
#             dot.node(name=str(id(n)) + n.operation, label=n.operation)
#             dot.edge(str(id(n)) + n.operation, str(id(n)))

#     for n1, n2 in edges:
#         dot.edge(str(id(n1)), str(id(n2)) + n2.operation)

#     return dot


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


import inspect

# color patterns
RED_PATTERN = '\033[31m%s\033[0m'
GREEN_PATTERN = '\033[32m%s\033[0m'
BLUE_PATTERN = '\033[34m%s\033[0m'
PEP_PATTERN = '\033[36m%s\033[0m'
BROWN_PATTERN = '\033[33m%s\033[0m'

def mprint(obj, magic_methods=True, private_methods=True, public_methods=True):
    # Split items based on their types
    if magic_methods:
        magic_methods = [x for x in dir(obj) if x.startswith("__") and x.endswith("__")]
        print("\n\033[93mMagic Methods:\033[0m")
        for item in sorted(magic_methods):
            print(f"    {item}")
    
    if private_methods:
        private_methods = [x for x in dir(obj) if x.startswith("_") and not x in magic_methods]
        print("\n\033[93mPrivate Methods:\033[0m")
        for item in sorted(private_methods):
            print(f"    {item}")
    
    if public_methods:
        public_methods = [x for x in dir(obj) if not x.startswith("_")]
        print("\n\033[93mPublic Methods:\033[0m")
        for item in sorted(public_methods):
            print(f"    {item}")


def cprint(expr, globals=None, locals=None):
    """
    Custom print function that prints the name of the variable/expression
    alongside its value.
    
    Parameters:
    - expr: The expression to evaluate.
    - globals, locals (dict, optional): Optional dictionaries to specify the namespace 
      for the evaluation. This allows the function to access variables outside of its 
      local scope.
    
    Example:
    x = 10
    cprint(x)
    # Output: x: 10
    """
    # Fetch the line of code that called cprint
    frame = inspect.currentframe().f_back
    # line = frame.f_code.co_filename, frame.f_lineno
    call_line = inspect.getframeinfo(frame).code_context[0].strip()

    # Extract the argument from the line
    arg_str = call_line[call_line.index('(') + 1:-1].strip()

    try:
        value = expr
        # print(f"{arg_str}: {value}")
        print(f"\033[93m{arg_str}\033[0m: \n{value}\n")
    except Exception as e:
        print(f"Error evaluating {arg_str}: {e}")


def sprint(expr, globals=None, locals=None):
    """
    Custom print function that prints the name of the variable/expression
    alongside its value.

    Parameters:
    - expr (str): The expression to evaluate as a string.
    - globals, locals (dict, optional): Optional dictionaries to specify the namespace
      for the evaluation. This allows the function to access variables outside of its
      local scope.

    Example:
    x = 10
    cprint_str("x")
    # Output: x: 10
    """
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
