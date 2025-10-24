#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.

"""
Utilities for loading template context from YAML configuration files.

This module provides functions to load Mako template context from YAML files
associated with interface specifications. This allows externalizing template
variables (like operator lists, kernel parameters, etc.) from Python source code
to declarative YAML configuration.

YAML files should be named <interface>.yml and located in the interfaces directory.
Context from YAML files is merged with context passed programmatically, with
programmatic context taking precedence.

Example:
    operators.yml:
        operators:
          - [add, "+"]
          - [sub, "-"]
          - [mul, "*"]
          - [divop, "/"]
"""

import importlib
import os
import yaml
from typing import Any, Callable, Dict

from yapocis.interfaces import find_interface

# Type alias for context generator functions
ContextGenerator = Callable[[Dict[str, Any]], Dict[str, Any]]


def load_interface_context(interface_name: str) -> Dict[str, Any]:
    """
    Load template context from YAML file for given interface.

    Looks for <interface_name>.yml in the interfaces directory.
    Returns empty dict if YAML file doesn't exist.

    Args:
        interface_name: Name of interface (e.g., "operators")

    Returns:
        Dictionary of template context variables, or empty dict if no YAML file

    Example:
        >>> context = load_interface_context("operators")
        >>> context.get("operators")
        [["add", "+"], ["sub", "-"], ["mul", "*"], ["divop", "/"]]
    """
    # Find the interface file path
    interface_path = find_interface(interface_name)
    if interface_path is None:
        return {}

    # Look for YAML file in same directory
    interface_dir = os.path.dirname(interface_path)
    yaml_path = os.path.join(interface_dir, f"{interface_name}.yml")

    if not os.path.exists(yaml_path):
        return {}

    # Load YAML context
    try:
        with open(yaml_path, 'r') as f:
            context = yaml.safe_load(f)
            if context is None:
                return {}
            if not isinstance(context, dict):
                raise ValueError(
                    f"YAML file {yaml_path} must contain a dictionary at top level, "
                    f"got {type(context).__name__}"
                )
            return context
    except yaml.YAMLError as e:
        raise RuntimeError(
            f"Failed to parse YAML context from {yaml_path}: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to load YAML context from {yaml_path}: {e}"
        ) from e


def merge_context(yaml_context: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    """
    Merge YAML context with programmatically-passed context.

    Programmatic context takes precedence over YAML context. This allows
    overriding YAML defaults when needed.

    Args:
        yaml_context: Context loaded from YAML file
        **kwargs: Programmatically-passed context variables

    Returns:
        Merged context dictionary (YAML values + kwargs overrides)

    Example:
        >>> yaml_ctx = {"operators": [["add", "+"]], "precision": "float"}
        >>> merged = merge_context(yaml_ctx, precision="double")
        >>> merged["precision"]
        'double'
        >>> merged["operators"]
        [['add', '+']]
    """
    merged = dict(yaml_context)
    merged.update(kwargs)
    return merged


def get_interface_context(interface_name: str, **kwargs: Any) -> Dict[str, Any]:
    """
    Get complete template context for an interface.

    Loads YAML context (if exists), applies context generator if specified,
    and merges with programmatic context. Programmatic context takes precedence.

    Args:
        interface_name: Name of interface (e.g., "operators")
        **kwargs: Additional context variables (override YAML)

    Returns:
        Complete merged context dictionary

    Example:
        >>> # With operators.yml defining default operators
        >>> context = get_interface_context("operators")
        >>> # Override operators list programmatically
        >>> context = get_interface_context("operators",
        ...     operators=[["add", "+"], ["sub", "-"]])
    """
    yaml_context = load_interface_context(interface_name)

    # Apply context generator if specified in YAML
    yaml_context = apply_context_generator(yaml_context)

    return merge_context(yaml_context, **kwargs)


def apply_context_generator(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply context generator if specified in metadata.

    Looks for '_generator' key in context and invokes the specified
    Python function to transform the context. The generator function
    is loaded dynamically from the specified module.

    Args:
        context: YAML context dictionary

    Returns:
        Transformed context dictionary (with _generator metadata removed)

    Raises:
        ValueError: If _generator format is invalid
        ImportError: If generator module cannot be imported
        AttributeError: If generator function not found in module

    Example:
        ```yaml
        # convolvesep.yml
        _generator:
          module: yapocis.lib.gaussian
          function: generate_convolvesep_context

        gaussian_params:
          max_width: 150
          scale: 1.6
        ```

        The function `yapocis.lib.gaussian.generate_convolvesep_context(context)`
        will be called to transform the context before template rendering.
    """
    if '_generator' not in context:
        return context

    # Extract and remove metadata
    gen_spec = context.pop('_generator')

    # Validate generator spec
    if not isinstance(gen_spec, dict):
        raise ValueError(f"_generator must be a dict, got {type(gen_spec).__name__}")

    if 'module' not in gen_spec or 'function' not in gen_spec:
        raise ValueError("_generator must have 'module' and 'function' keys")

    module_name = gen_spec['module']
    function_name = gen_spec['function']

    # Import module and get function
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"Cannot import generator module '{module_name}': {e}"
        ) from e

    try:
        generator_func = getattr(module, function_name)
    except AttributeError as e:
        raise AttributeError(
            f"Module '{module_name}' has no function '{function_name}': {e}"
        ) from e

    # Invoke generator
    return generator_func(context)
