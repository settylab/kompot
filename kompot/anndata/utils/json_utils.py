"""JSON serialization utilities for AnnData objects."""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from anndata import AnnData
import logging
import json

logger = logging.getLogger("kompot")


def jsonable_encoder(obj: Any) -> Any:
    """
    Recursively convert complex nested data structures to JSON-serializable format.
    
    Parameters
    ----------
    obj : Any
        The object to convert. Can be a nested structure of dictionaries, lists, 
        numpy arrays, etc.
        
    Returns
    -------
    Any
        A JSON-serializable representation of the input object.
    """
    if isinstance(obj, dict):
        # Handle dictionaries recursively
        return {key: jsonable_encoder(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        # Handle lists recursively
        return [jsonable_encoder(item) for item in obj]
    elif isinstance(obj, tuple):
        # Convert tuples to lists and handle recursively
        return [jsonable_encoder(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        # Convert numpy arrays to lists
        if obj.ndim == 0:
            return obj.item()  # Convert 0-d arrays to scalars
        else:
            return obj.tolist()
    elif isinstance(obj, np.integer):
        # Convert numpy integer types to Python int
        return int(obj)
    elif isinstance(obj, np.floating):
        # Convert numpy floating types to Python float
        return float(obj)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        # These types are already JSON-serializable
        return obj
    else:
        # Convert anything else to a string
        return str(obj)


def to_json_string(obj: Any) -> str:
    """
    Convert any Python object to a JSON string.
    
    Parameters
    ----------
    obj : Any
        The object to convert to a JSON string.
        
    Returns
    -------
    str
        JSON string representation of the object.
    """
    return json.dumps(jsonable_encoder(obj))


def from_json_string(json_str: str) -> Any:
    """
    Convert a JSON string back to a Python object.
    
    Parameters
    ----------
    json_str : str
        JSON string to convert.
        
    Returns
    -------
    Any
        Python object representation of the JSON string.
    """
    if not isinstance(json_str, str):
        return json_str  # If it's not a string, return as is
        
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Error decoding JSON string: {e}")
        return json_str  # Return the original string if it's not valid JSON


def get_json_metadata(adata: AnnData, key_path: str) -> Any:
    """
    Retrieve and deserialize metadata from AnnData.uns with automatic JSON parsing.
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing the metadata.
    key_path : str
        Dot-separated path to the metadata key (e.g., "kompot_da.run_history").
        
    Returns
    -------
    Any
        The deserialized metadata, or None if not found.
    """
    # Split the key path
    keys = key_path.split(".")
    
    # Navigate to the final container
    current = adata.uns
    for key in keys[:-1]:
        if key not in current:
            return None
        current = current[key]
    
    # Get the final value
    final_key = keys[-1]
    if final_key not in current:
        return None
    
    value = current[final_key]
    
    # If it's a string, try to deserialize it
    if isinstance(value, str):
        try:
            return from_json_string(value)
        except Exception as e:
            logger.warning(f"Error deserializing metadata at {key_path}: {e}")
            return value
    
    # For lists and dicts, recursively deserialize any string elements that might be JSON
    if isinstance(value, list):
        # Create a new list with potentially deserialized items
        result = []
        for item in value:
            if isinstance(item, str):
                try:
                    item = from_json_string(item)
                except Exception:
                    pass  # Keep the original string if it's not valid JSON
            result.append(item)
        return result
    elif isinstance(value, dict):
        # Create a new dict with potentially deserialized values
        result = {}
        for k, v in value.items():
            if isinstance(v, str):
                try:
                    v = from_json_string(v)
                except Exception:
                    pass  # Keep the original string if it's not valid JSON
            result[k] = v
        return result
    
    return value


def set_json_metadata(adata: AnnData, key_path: str, value: Any) -> bool:
    """
    Store metadata in AnnData.uns with automatic JSON serialization for complex objects.
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object to store metadata in.
    key_path : str
        Dot-separated path to the metadata key (e.g., "kompot_da.run_history").
    value : Any
        The value to store. Complex objects will be serialized to JSON.
        
    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    # Split the key path
    keys = key_path.split(".")
    
    # Navigate to the final container, creating intermediate dictionaries if needed
    current = adata.uns
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        if not isinstance(current[key], dict):
            logger.warning(f"Cannot navigate to {key} in {current}, not a dictionary")
            return False
        current = current[key]
    
    # Store the value, serializing if complex
    final_key = keys[-1]
    
    # Check if the value is a complex object that needs serialization
    if isinstance(value, (dict, list, tuple)) or (
        isinstance(value, np.ndarray) and value.size > 1
    ):
        current[final_key] = to_json_string(value)
    else:
        # Simple values can be stored directly
        current[final_key] = value
    
    return True