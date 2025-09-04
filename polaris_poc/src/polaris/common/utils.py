import ast
import random
from typing import Any


def jittered_backoff(attempt: int, base: float, max_delay: float) -> float:
    """Calculate exponential backoff with jitter.
    
    Args:
        attempt: Current attempt number (0-based)
        base: Base delay in seconds
        max_delay: Maximum delay in seconds
    
    Returns:
        Delay in seconds with full jitter
    """
    exp = min(max_delay, base * (2 ** attempt))
    # full jitter
    return random.uniform(0, exp)

def safe_eval(expr: str, variables: dict) -> Any:
    """
    Safely evaluate a mathematical expression using limited AST nodes.
    This function parses the given expression, restricts the AST to a set of allowed nodes,
    and then evaluates the expression with the provided variables. It is intended to perform
    a safe evaluation by preventing execution of potentially unsafe code.
    
    Args:
        expr (str): The expression to evaluate.
        variables (dict): A dictionary containing variable names and their corresponding values.
    Returns:
        The result of the evaluated expression.
    Raises:
        ValueError: If the expression contains any AST nodes that are not explicitly allowed.
    """
    # Define allowed AST nodes
    allowed_nodes = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Load,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
        ast.USub, ast.UAdd, ast.Name, ast.BoolOp, ast.Or, ast.And, 
        ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.Call, ast.Attribute
    }

    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if type(node) not in allowed_nodes:
            raise ValueError(f"Disallowed expression node: {type(node).__name__}")
    compiled = compile(tree, filename="<ast>", mode="eval")
    return eval(compiled, {"__builtins__": {}}, variables)
