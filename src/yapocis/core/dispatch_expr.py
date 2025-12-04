"""
dispatch_expr.py - Expression evaluator for dispatch_size trait

Evaluates expressions like:
- "height_of input / 2"
- "width_of input"
- "sizeof output"
- "64"

Used by kernel dispatch_size trait to compute global_size at runtime.
"""

import re
from typing import Any

import numpy as np


def evaluate_dispatch_expression(expr: str, args: tuple[Any, ...], params: list) -> int:
    """
    Evaluate a dispatch_size expression.

    Args:
        expr: Expression string like "height_of input / 2"
        args: Tuple of actual kernel arguments (numpy arrays, scalars)
        params: List of parameter specs from interface

    Returns:
        Evaluated integer dimension

    Examples:
        >>> evaluate_dispatch_expression("height_of input / 2", (arr,), params)
        112  # if arr.shape[1] == 224

        >>> evaluate_dispatch_expression("width_of input", (arr,), params)
        64  # if arr.shape[0] == 64
    """
    expr = expr.strip()

    # Try to parse as integer literal first
    try:
        return int(expr)
    except ValueError:
        pass

    # Try to parse as float literal and convert to int
    try:
        return int(float(expr))
    except ValueError:
        pass

    # Parse dimension reference: "dimension_of parameter"
    # e.g., "height_of input", "width_of output", "sizeof input", "dim_0_of input"
    dim_pattern = r"(width_of|height_of|depth_of|sizeof|dim_\d+_of)\s+(\w+)"
    match = re.search(dim_pattern, expr)

    if match:
        dim_type = match.group(1)
        param_name = match.group(2)

        # Find the parameter and its corresponding argument
        param_idx = None
        for i, param in enumerate(params):
            if param.name == param_name:
                param_idx = i
                break

        if param_idx is None or param_idx >= len(args):
            raise ValueError(f"Cannot find parameter '{param_name}' in arguments")

        arg = args[param_idx]

        if not isinstance(arg, np.ndarray):
            raise ValueError(
                f"Parameter '{param_name}' must be a numpy array for {dim_type}"
            )

        # Get the dimension value
        # dim_N_of: Direct access to shape[N]
        if dim_type.startswith("dim_"):
            # Extract N from "dim_N_of"
            n = int(dim_type.split("_")[1])
            if n >= len(arg.shape):
                raise ValueError(
                    f"Array '{param_name}' does not have dimension {n} (shape={arg.shape})"
                )
            dim_value = arg.shape[n]
        # Traditional dimension specifiers (for backward compatibility)
        # For yapocis graphics arrays in (H, W) format:
        # - width_of → shape[0]
        # - height_of → shape[1] if ndim > 1, else 1
        # For yapocis CNN arrays in (C, H, W) format:
        # - depth_of (channels) → shape[0]
        # - height_of → shape[1]
        # - width_of → shape[2]
        elif dim_type == "width_of":
            dim_value = arg.shape[2] if len(arg.shape) > 2 else arg.shape[0]
        elif dim_type == "height_of":
            dim_value = arg.shape[1] if len(arg.shape) > 1 else 1
        elif dim_type == "depth_of":
            dim_value = arg.shape[0]
        elif dim_type == "sizeof":
            dim_value = arg.size
        else:
            raise ValueError(f"Unknown dimension type: {dim_type}")

        # Replace the dimension reference with its value in the expression
        expr = expr.replace(match.group(0), str(dim_value))

    # Now evaluate the arithmetic expression
    # Support: +, -, *, /, //, %
    # This is safe because we've already resolved all references to numbers
    try:
        # Use eval with restricted namespace for safety
        result = eval(expr, {"__builtins__": {}}, {})
        return int(result)
    except Exception as e:
        raise ValueError(f"Cannot evaluate dispatch expression '{expr}': {e}")


def evaluate_dispatch_size(
    dispatch_size: list[str] | tuple[str, ...], args: tuple[Any, ...], params: list
) -> tuple[int, ...]:
    """
    Evaluate a dispatch_size tuple expression.

    Args:
        dispatch_size: Tuple of expression strings, e.g.,
                      ("height_of input / 2", "width_of input / 2", "width_of input")
        args: Tuple of actual kernel arguments
        params: List of parameter specs from interface

    Returns:
        Tuple of evaluated dimensions for global_size

    Examples:
        >>> evaluate_dispatch_size(
        ...     ("height_of input / 2", "width_of input / 2", "width_of input"),
        ...     (input_arr, output_arr),
        ...     params
        ... )
        (112, 112, 64)  # for input shape (64, 224, 224)
    """
    result = []
    for expr in dispatch_size:
        if isinstance(expr, (int, float)):
            result.append(int(expr))
        else:
            result.append(evaluate_dispatch_expression(str(expr), args, params))
    return tuple(result)
