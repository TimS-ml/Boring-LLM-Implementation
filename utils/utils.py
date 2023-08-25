import inspect

def cprint(expr, _globals=None, _locals=None):
    """
    Custom print function that prints the name of the variable/expression
    alongside its value.
    
    Parameters:
    - expr: The expression to evaluate.
    - _globals, _locals (dict, optional): Optional dictionaries to specify the namespace 
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


def cprint_str(expr, globals=None, locals=None):
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
