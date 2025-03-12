"""Core utility functions for anndata module."""

def _sanitize_name(name):
    """Convert a string to a valid column/key name.

    Args:
        name: String to convert.

    Returns:
        String with invalid characters replaced.
    """
    return "".join([c if c.isalnum() else "_" for c in name])