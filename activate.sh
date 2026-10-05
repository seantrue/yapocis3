#!/usr/bin/env bash
# Activate the virtualenv for yapocis3.
#
# yapocis3 requires Python >=3.10 (see pyproject.toml) and uses PyOpenCL; the
# matching interpreter + deps live in the virtualenvwrapper env
# ``~/.virtualenvs/yapocis-310`` (Python 3.10.12). The sibling ``yapocis`` env
# is Python 3.7 for the older bare-yapocis project and is NOT for this repo.
#
# Usage — SOURCE this, don't execute it:
#     source activate.sh        # or:  . ./activate.sh
# Deactivate with the usual ``deactivate``.

VENV="${YAPOCIS3_VENV:-$HOME/.virtualenvs/yapocis-310}"

if [ ! -f "$VENV/bin/activate" ]; then
    echo "yapocis3: virtualenv not found at $VENV" >&2
    echo "  set YAPOCIS3_VENV to override, or recreate it (python3.10 -m venv)." >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "yapocis3: activated $(python --version 2>&1) at $VIRTUAL_ENV"
