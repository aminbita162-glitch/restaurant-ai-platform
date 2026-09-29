# This module is a thin delegation shim.
# All persistence logic lives in core/persistence.py.
# Callers that already import from here continue to work unchanged.

from .core.persistence import (  # noqa: F401  re-exported for backwards compatibility
    PERSISTENCE_ERROR_MISSING_TENANT,
    init_db,
    save_run,
    get_last_run,
)
