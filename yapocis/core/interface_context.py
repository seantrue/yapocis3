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

import os
import yaml
from typing import Any, Dict

from yapocis.interfaces import find_interface


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

    Loads YAML context (if exists) and merges with programmatic context.
    Programmatic context takes precedence.

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
    return merge_context(yaml_context, **kwargs)
