#!/usr/bin/env bash
# Activate the virtualenv for yapocis3.
#
# The environment is the repository's uv-managed ``.venv`` (Python 3.13),
# built from pyproject.toml / uv.lock, with yapocis installed editable:
#     uv sync --python 3.13 --extra opencl --extra cuda --extra numba --group dev
# (``make install-linux`` does the same with the default interpreter.)
#
# If a CUDA toolkit is installed at /usr/local/cuda, its bin directory is put
# on PATH: pycuda compiles kernels with nvcc, and non-interactive shells
# (e.g. ``ssh host 'cmd'``) don't read ~/.bashrc.
#
# Usage — SOURCE this, don't execute it:
#     source activate.sh        # or:  . ./activate.sh
# Deactivate with the usual ``deactivate``.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${YAPOCIS3_VENV:-$REPO/.venv}"

if [ ! -f "$VENV/bin/activate" ]; then
    echo "yapocis3: virtualenv not found at $VENV" >&2
    echo "  set YAPOCIS3_VENV to override, or create it with uv sync (see above)." >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

if [ -x /usr/local/cuda/bin/nvcc ]; then
    export CUDA_HOME=/usr/local/cuda
    case ":$PATH:" in
        *":$CUDA_HOME/bin:"*) ;;
        *) export PATH="$CUDA_HOME/bin:$PATH" ;;
    esac
fi

echo "yapocis3: activated $(python --version 2>&1) at $VIRTUAL_ENV"
