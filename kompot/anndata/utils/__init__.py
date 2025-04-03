"""Utility functions and classes for working with AnnData objects."""

# JSON utilities
from .json_utils import (
    jsonable_encoder,
    to_json_string,
    from_json_string,
    get_json_metadata,
    set_json_metadata,
)

# Field tracking utilities
from .field_tracking import (
    get_run_history,
    append_to_run_history,
    get_last_run_info,
    generate_output_field_names,
    get_environment_info,
    detect_output_field_overwrite,
    _sanitize_name,
    validate_field_run_id,
    get_run_from_history,
)

# Group utilities
from .group_utils import (
    parse_groups,
    check_underrepresentation,
    refine_filter_for_underrepresentation,
    apply_cell_filter,
)

# Run info classes
from .runinfo import (
    RunInfo,
    RunComparison,
)

__all__ = [
    # JSON utilities
    'jsonable_encoder',
    'to_json_string',
    'from_json_string',
    'get_json_metadata',
    'set_json_metadata',
    
    # Field tracking utilities
    'get_run_history',
    'append_to_run_history',
    'get_last_run_info',
    'generate_output_field_names',
    'get_environment_info',
    'detect_output_field_overwrite',
    '_sanitize_name',
    'validate_field_run_id',
    'get_run_from_history',
    
    # Group utilities
    'parse_groups',
    'check_underrepresentation',
    'refine_filter_for_underrepresentation',
    'apply_cell_filter',
    
    # Run info classes
    'RunInfo',
    'RunComparison',
]