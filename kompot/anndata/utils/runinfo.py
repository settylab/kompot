"""Classes for managing run information and comparisons."""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from anndata import AnnData
import logging
import copy
import pprint

from .json_utils import from_json_string, get_json_metadata, set_json_metadata
from .field_tracking import get_run_from_history, validate_field_run_id

logger = logging.getLogger("kompot")


class RunInfo:
    """
    Class for accessing run information for differential analysis.
    
    Provides access to run history, parameters, and result fields.
    
    Attributes
    ----------
    adata : AnnData
        AnnData object containing the run history
    run_id : int
        Requested run ID (may be negative for relative indexing)
    adjusted_run_id : int
        Actual run ID after adjusting for negative indexing
    analysis_type : str
        Type of analysis: 'de' for differential expression or 'da' for differential abundance
    storage_key : str
        Key for accessing the analysis data in adata.uns
    run_info : dict
        Dictionary with all information about the run
    field_names : dict
        Dictionary with field names used in this run
    params : dict
        The parameters used for this analysis
    environment : dict
        Information about the environment where the analysis was run
    overwritten_fields : list
        List of fields that were overwritten by newer runs
    missing_fields : list
        List of fields that are missing/deleted from the AnnData object
    """
    
    def __init__(self, 
                 adata, 
                 run_id: Optional[int] = None, 
                 analysis_type: Optional[str] = None):
        """
        Initialize a RunInfo object.
        
        Parameters
        ----------
        adata : AnnData
            AnnData object containing run history
        run_id : int, optional
            Run ID to retrieve. Negative indices count from the end.
            If None, uses the most recent run (-1).
        analysis_type : str, optional
            Type of analysis: 'de' for differential expression or 
            'da' for differential abundance. If None, attempts to detect.
        """
        self.adata = adata
        if run_id is None:
            run_id = -1  # Default to most recent run
        self.run_id = run_id
        
        # Detect analysis type if not provided
        if analysis_type is None:
            # Try to detect from uns keys
            if 'kompot_de' in adata.uns and 'run_history' in adata.uns['kompot_de']:
                analysis_type = 'de'
            elif 'kompot_da' in adata.uns and 'run_history' in adata.uns['kompot_da']:
                analysis_type = 'da'
            else:
                raise ValueError("Could not detect analysis type. Please specify 'de' or 'da'.")
                
        if analysis_type not in ['de', 'da']:
            raise ValueError(f"Invalid analysis_type: {analysis_type}. Must be 'de' or 'da'.")
            
        self.analysis_type = analysis_type
        self.storage_key = f"kompot_{analysis_type}"
        
        # Check if run history exists
        if (self.storage_key not in adata.uns or 
            'run_history' not in adata.uns[self.storage_key] or
            len(adata.uns[self.storage_key]['run_history']) == 0):
            raise ValueError(f"No run history found for {analysis_type} analysis.")
        
        # Get run info
        self.run_info = get_run_from_history(adata, run_id=run_id, analysis_type=analysis_type)
        
        if self.run_info is None:
            raise ValueError(f"Run ID {run_id} not found in {analysis_type} run history.")
            
        # Set adjusted run_id
        self.adjusted_run_id = self.run_info.get('adjusted_run_id', None)
        
        # Extract key information
        self.field_names = self.run_info.get('field_names', {})
        self.params = self.run_info.get('params', {}).copy()  # Make a copy to avoid modifying the original
        self.environment = self.run_info.get('environment', {})
        self.timestamp = self.run_info.get('timestamp', '')
        
        # Ensure result_key is included in params if missing
        if 'result_key' not in self.params and 'result_key' in self.run_info:
            self.params['result_key'] = self.run_info['result_key']
        
        # Get all fields modified by this run
        self.adata_fields = self._get_fields_for_run()

        # Check for fields that have been overwritten by newer runs
        self.overwritten_fields = self._check_overwritten_fields()

        # Check for fields that are missing/deleted
        self.missing_fields = self._check_missing_fields()
        
    def _get_fields_for_run(self) -> Dict[str, List[str]]:
        """
        Get all fields in the AnnData object that were written by this run.
        
        Returns
        -------
        Dict[str, List[str]]
            Dictionary with AnnData locations as keys and lists of field names as values
        """
        result = {}
        
        # Get fields from field_mapping in the run_info - this is the only source of truth
        run_data = self.get_raw_data()
        field_mapping = run_data.get('field_mapping', {})
        
        # If field_mapping is a string, try to parse it as JSON
        if isinstance(field_mapping, str):
            try:
                field_mapping = from_json_string(field_mapping)
            except Exception as e:
                logger.warning(f"Error parsing field_mapping as JSON: {e}")
                field_mapping = {}
        
        if not field_mapping:
            logger.warning(f"No field_mapping found for run {self.adjusted_run_id}.")
            return {}
            
        # Initialize result - ensure mapping is a dict before accessing
        locations = set()
        for mapping_value in field_mapping.values():
            if isinstance(mapping_value, dict) and 'location' in mapping_value:
                locations.add(mapping_value['location'])
            elif isinstance(mapping_value, str):
                # Try to parse as JSON
                try:
                    mapping_dict = from_json_string(mapping_value)
                    if isinstance(mapping_dict, dict) and 'location' in mapping_dict:
                        locations.add(mapping_dict['location'])
                except Exception:
                    pass
                    
        for location in locations:
            result[location] = []
            
        # Add fields to their locations
        for field, mapping in field_mapping.items():
            # Handle case where mapping is a string (serialized JSON)
            if isinstance(mapping, str):
                try:
                    mapping = from_json_string(mapping)
                except Exception:
                    continue
                    
            # Only process dictionary mappings with location
            if isinstance(mapping, dict) and 'location' in mapping:
                location = mapping['location']
                if location not in result:
                    result[location] = []
                result[location].append(field)
        
        # Sort field lists for consistent display
        for location in result:
            result[location].sort()
            
        return result
    
    def _check_overwritten_fields(self) -> List[Dict[str, Any]]:
        """
        Check if any fields from this run have been overwritten by newer runs.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries with overwritten field information, each containing:
            - field: The field name
            - location: The location in AnnData (obs, var, etc.)
            - current_run_id: The run ID that now owns this field
            - expected_run_id: The run ID that should own this field (this run)
        """
        if (self.storage_key not in self.adata.uns or 
            'anndata_fields' not in self.adata.uns[self.storage_key]):
            return []
            
        tracking = self.adata.uns[self.storage_key]['anndata_fields']
        
        # If tracking is a JSON string, deserialize it
        if isinstance(tracking, str):
            tracking = from_json_string(tracking)
            
        overwritten = []
        
        # Get fields from field_mapping as the source of truth
        field_mapping = self.get_raw_data().get('field_mapping', {})
        
        # If field_mapping is a string, try to parse it as JSON
        if isinstance(field_mapping, str):
            try:
                field_mapping = from_json_string(field_mapping)
            except Exception as e:
                logger.warning(f"Error parsing field_mapping as JSON: {e}")
                field_mapping = {}
        
        # If no field_mapping, we don't know what fields to check
        if not field_mapping:
            logger.warning(f"No field_mapping found for run {self.adjusted_run_id} to check for overwritten fields.")
            return []
        
        # Check each field from field_mapping against the tracking info
        for field, mapping in field_mapping.items():
            # Handle case where mapping might be a JSON string
            if isinstance(mapping, str):
                try:
                    mapping = from_json_string(mapping)
                except Exception:
                    continue
                    
            # Only process dictionary mappings
            if not isinstance(mapping, dict):
                continue
                
            location = mapping.get('location')
            if not location or location not in tracking or field not in tracking[location]:
                # Skip fields not in the tracking dictionary
                continue
                
            # Check if the field is attributed to a different run
            current_run_id = tracking[location][field]
            if current_run_id != self.adjusted_run_id:
                overwritten.append({
                    'field': field,
                    'location': location,
                    'current_run_id': current_run_id,
                    'expected_run_id': self.adjusted_run_id
                })

        return overwritten

    def _check_missing_fields(self) -> List[Dict[str, Any]]:
        """
        Check if any fields from this run are missing/deleted from the AnnData object.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries with missing field information, each containing:
            - field: The field name
            - location: The location in AnnData (obs, var, layers, obsp, varm)
            - type: The field type (e.g., 'imputed', 'fold_change', etc.)
            - description: Description of the field
        """
        missing = []

        # Get fields from field_mapping as the source of truth
        field_mapping = self.get_raw_data().get('field_mapping', {})

        # If field_mapping is a string, try to parse it as JSON
        if isinstance(field_mapping, str):
            try:
                field_mapping = from_json_string(field_mapping)
            except Exception as e:
                logger.warning(f"Error parsing field_mapping as JSON: {e}")
                field_mapping = {}

        # If no field_mapping, we don't know what fields to check
        if not field_mapping:
            return []

        # Check each field from field_mapping
        for field, mapping in field_mapping.items():
            # Handle case where mapping might be a JSON string
            if isinstance(mapping, str):
                try:
                    mapping = from_json_string(mapping)
                except Exception:
                    continue

            # Only process dictionary mappings
            if not isinstance(mapping, dict):
                continue

            location = mapping.get('location')
            if not location:
                continue

            # Check if field exists in the AnnData object
            field_exists = False
            if location == 'var':
                field_exists = field in self.adata.var.columns
            elif location == 'obs':
                field_exists = field in self.adata.obs.columns
            elif location == 'layers':
                field_exists = field in self.adata.layers
            elif location == 'obsp':
                field_exists = field in self.adata.obsp
            elif location == 'varm':
                field_exists = field in self.adata.varm
            elif location == 'uns':
                field_exists = field in self.adata.uns

            if not field_exists:
                missing.append({
                    'field': field,
                    'location': location,
                    'type': mapping.get('type', 'unknown'),
                    'description': mapping.get('description', '')
                })

        return missing

    def compare_with(self, other_run_id: int) -> 'RunComparison':
        """
        Compare this run with another run.
        
        Parameters
        ----------
        other_run_id : int
            Run ID to compare with
            
        Returns
        -------
        RunComparison
            Object containing comparison results with nice display methods
        """
        return RunComparison(self.adata, self.run_id, other_run_id, self.analysis_type)
    
    def get_data(self) -> Dict[str, Any]:
        """
        Get all data related to this run.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with all run data
        """
        # Get field data based on adata_fields
        field_data = {}
        
        # Initialize adata_fields as empty dict if not present
        if not hasattr(self, 'adata_fields') or not self.adata_fields:
            self.adata_fields = {}
            
        for location, fields in self.adata_fields.items():
            field_data[location] = {}
            
            if location == 'obs':
                for field in fields:
                    if field in self.adata.obs:
                        field_data[location][field] = self.adata.obs[field]
            elif location == 'var':
                for field in fields:
                    if field in self.adata.var:
                        field_data[location][field] = self.adata.var[field]
            elif location == 'uns':
                for field in fields:
                    if field in self.adata.uns:
                        field_data[location][field] = self.adata.uns[field]
            elif location == 'layers':
                for field in fields:
                    if field in self.adata.layers:
                        field_data[location][field] = self.adata.layers[field]
        
        return {
            'run_id': self.run_id,
            'adjusted_run_id': self.adjusted_run_id,
            'analysis_type': self.analysis_type,
            'field_names': self.field_names,
            'params': self.params,
            'environment': self.environment,
            'timestamp': self.timestamp,
            'overwritten_fields': self.overwritten_fields,
            'field_data': field_data
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of this run with key information.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with run summary
        """
        # Get basic information without field data
        summary = {
            'run_id': self.run_id,
            'adjusted_run_id': self.adjusted_run_id,
            'analysis_type': self.analysis_type,
            'timestamp': self.timestamp,
            'conditions': f"{self.params.get('condition1', 'unknown')} to {self.params.get('condition2', 'unknown')}",
            'obsm_key': self.params.get('obsm_key', 'unknown'),
            'layer': self.params.get('layer', None),
            'uses_sample_variance': self.params.get('use_sample_variance', False),
            'field_count': sum(len(fields) for fields in self.adata_fields.values()) if self.adata_fields else 0,
            'overwritten_field_count': len(self.overwritten_fields) if hasattr(self, 'overwritten_fields') else 0,
            'overwritten_fields': self.overwritten_fields if hasattr(self, 'overwritten_fields') else [],
            'missing_field_count': len(self.missing_fields) if hasattr(self, 'missing_fields') else 0,
            'missing_fields': self.missing_fields if hasattr(self, 'missing_fields') else []
        }
        
        # Add group information if available
        raw_data = self.get_raw_data()
        has_groups = raw_data.get('has_groups', False)
        if has_groups:
            groups_summary = raw_data.get('groups_summary', {})
            summary['has_groups'] = True
            summary['groups_count'] = groups_summary.get('count', 0)
            groups_names = groups_summary.get('names', [])
            # Provide a short preview of group names
            group_names_preview = ", ".join(groups_names[:3])
            if len(groups_names) > 3:
                group_names_preview += f" and {len(groups_names) - 3} more"
            summary['groups'] = group_names_preview
        else:
            summary['has_groups'] = False
        
        # Don't add anndata_locations directly to summary
        # We'll use it to enhance field listings instead
        return summary
    
    def get_raw_data(self) -> Dict[str, Any]:
        """
        Get the raw run info data without any processing.
        
        Returns
        -------
        Dict[str, Any]
            The raw run info data
        """
        return self.run_info
    
    def __repr__(self) -> str:
        """
        String representation of the RunInfo object.
        
        Returns
        -------
        str
            String representation
        """
        summary = self.get_summary()
        return f"RunInfo(analysis_type={summary['analysis_type']}, run_id={summary['adjusted_run_id']}, timestamp={summary['timestamp']})"
        
    def _repr_html_(self) -> str:
        """
        HTML representation for Jupyter notebooks.
        
        Returns
        -------
        str
            HTML representation
        """
        summary = self.get_summary()
        
        # CSS styles for better formatting and interactivity
        css_styles = """
        <style>
            .kompot-runinfo {
                max-width: 900px;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
            }
            .kompot-runinfo h3 {
                margin-bottom: 10px;
                font-weight: 600;
            }
            .kompot-runinfo details {
                margin-top: 15px;
                margin-bottom: 10px;
            }
            .kompot-runinfo summary {
                color: #555;
                cursor: pointer;
                font-weight: 500;
                font-size: 1.1em;
                padding: 5px 0;
                list-style: none;
                user-select: none;
            }
            .kompot-runinfo summary::-webkit-details-marker {
                display: none;
            }
            .kompot-runinfo summary::before {
                content: "▶ ";
                display: inline-block;
                font-size: 0.8em;
                color: #888;
                margin-right: 5px;
            }
            .kompot-runinfo details[open] > summary::before {
                content: "▼ ";
            }
            .kompot-runinfo table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 15px;
            }
            .kompot-runinfo th {
                background-color: rgba(0, 0, 0, 0.04);
                text-align: left;
                padding: 8px;
                border: 1px solid rgba(0, 0, 0, 0.12);
                font-weight: 500;
            }
            .kompot-runinfo td {
                padding: 6px 8px;
                border: 1px solid rgba(0, 0, 0, 0.12);
                vertical-align: top;
            }
            .kompot-runinfo tr {
                background-color: transparent;
            }
            .kompot-runinfo tr:nth-child(even) {
                background-color: rgba(0, 0, 0, 0.02);
            }
            .kompot-runinfo .overwritten {
                color: #c62828;
                font-weight: 600;
            }
            .kompot-runinfo .active {
                color: #2e7d32;
            }
            .kompot-runinfo .field-section .field-row:hover {
                background-color: rgba(0, 0, 0, 0.04);
            }
            .kompot-runinfo .badge {
                display: inline-block;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.85em;
                font-weight: 500;
                margin-left: 5px;
            }
            .kompot-runinfo .badge-success {
                background-color: rgba(46, 125, 50, 0.12);
                color: #2e7d32;
                border: 1px solid rgba(46, 125, 50, 0.3);
            }
            .kompot-runinfo .badge-warning {
                background-color: rgba(237, 108, 2, 0.12);
                color: #e65100;
                border: 1px solid rgba(237, 108, 2, 0.3);
            }
            .kompot-runinfo .badge-danger {
                background-color: rgba(198, 40, 40, 0.12);
                color: #c62828;
                border: 1px solid rgba(198, 40, 40, 0.3);
            }
            .kompot-runinfo .summary-box {
                background-color: rgba(0, 0, 0, 0.02);
                border: 1px solid rgba(0, 0, 0, 0.08);
                padding: 8px 12px;
                margin: 10px 0;
                border-radius: 4px;
            }
            .kompot-runinfo .location-header {
                background-color: rgba(0, 0, 0, 0.04);
                font-weight: bold;
            }

            @media (prefers-color-scheme: dark) {
                .kompot-runinfo summary {
                    color: #b8b8b8;
                }
                .kompot-runinfo summary::before {
                    color: #888;
                }
                .kompot-runinfo th {
                    background-color: rgba(255, 255, 255, 0.06);
                    border-color: rgba(255, 255, 255, 0.15);
                }
                .kompot-runinfo td {
                    border-color: rgba(255, 255, 255, 0.15);
                }
                .kompot-runinfo tr:nth-child(even) {
                    background-color: rgba(255, 255, 255, 0.03);
                }
                .kompot-runinfo .field-section .field-row:hover {
                    background-color: rgba(255, 255, 255, 0.06);
                }
                .kompot-runinfo .overwritten {
                    color: #ef5350;
                }
                .kompot-runinfo .active {
                    color: #66bb6a;
                }
                .kompot-runinfo .badge-success {
                    background-color: rgba(102, 187, 106, 0.2);
                    color: #81c784;
                    border-color: rgba(102, 187, 106, 0.4);
                }
                .kompot-runinfo .badge-warning {
                    background-color: rgba(255, 167, 38, 0.2);
                    color: #ffb74d;
                    border-color: rgba(255, 167, 38, 0.4);
                }
                .kompot-runinfo .badge-danger {
                    background-color: rgba(239, 83, 80, 0.2);
                    color: #e57373;
                    border-color: rgba(239, 83, 80, 0.4);
                }
                .kompot-runinfo .summary-box {
                    background-color: rgba(255, 255, 255, 0.03);
                    border-color: rgba(255, 255, 255, 0.12);
                }
                .kompot-runinfo .location-header {
                    background-color: rgba(255, 255, 255, 0.08);
                }
            }
        </style>
        """
        
        # Build HTML
        html = [
            css_styles,
            "<div class='kompot-runinfo'>",
            f"<h3>Run {summary['adjusted_run_id']} ({summary['analysis_type'].upper()} Analysis)</h3>",

            # Summary Section
            "<details open>",
            "<summary>Run Summary</summary>",
            "<div class='summary-box'>",
            f"<strong>Analysis:</strong> {summary['analysis_type'].upper()} &nbsp;|&nbsp; ",
            f"<strong>Run ID:</strong> {summary['adjusted_run_id']} &nbsp;|&nbsp; ",
            f"<strong>Timestamp:</strong> {summary['timestamp']} &nbsp;|&nbsp; ",
            f"<strong>Conditions:</strong> {summary['conditions']}",
        ]
        
        # Add badges for field status
        if summary.get('missing_field_count', 0) > 0:
            html.append(f"<span class='badge badge-danger'>Fields Missing: {summary['missing_field_count']}</span>")
        if summary.get('overwritten_field_count', 0) > 0:
            html.append(f"<span class='badge badge-warning'>Fields Overwritten: {summary['overwritten_field_count']}</span>")
        if summary.get('overwritten_field_count', 0) == 0 and summary.get('missing_field_count', 0) == 0:
            html.append("<span class='badge badge-success'>All Fields Present</span>")

        html.append("</div>")  # Close summary box
        
        # Add basic parameter table
        html.append("<table>")
        html.append("<tr><th style='width:30%'>Parameter</th><th style='width:70%'>Value</th></tr>")
        
        # Key parameters to show in summary (always visible)
        key_params = ['conditions', 'obsm_key', 'result_key', 'uses_sample_variance', 'layer', 'timestamp']
        for k in key_params:
            if k in summary and summary[k] is not None:
                html.append(f"<tr><td>{k}</td><td>{summary[k]}</td></tr>")
        
        # Field counts
        html.append(f"<tr><td>Fields Created</td><td>{summary.get('field_count', 0)}</td></tr>")
        
        # Group information if available
        if summary.get('has_groups', False):
            html.append(f"<tr><td>Groups</td><td>{summary.get('groups', '')} ({summary.get('groups_count', 0)} total)</td></tr>")
        
        html.append("</table>")
        html.append("</details>")  # Close summary section

        # All Parameters Section (initially collapsed)
        html.append("<details>")
        html.append("<summary>All Parameters</summary>")
        html.append("<table>")
        html.append("<tr><th style='width:30%'>Parameter</th><th style='width:70%'>Value</th></tr>")
        
        # Add all parameters from params dictionary
        for k, v in sorted(self.params.items()):
            if v is not None:  # Only show non-None parameters
                # Format the value based on type
                if isinstance(v, bool):
                    val_str = str(v)
                elif isinstance(v, (list, tuple)) and len(v) > 10:
                    val_str = f"{str(v[:10])[:-1]}, ... ({len(v)} items total)]"
                else:
                    val_str = str(v)
                html.append(f"<tr><td>{k}</td><td>{val_str}</td></tr>")
        
        html.append("</table>")
        html.append("</details>")  # Close parameters section

        # Environment Section (initially collapsed)
        if self.environment:
            html.append("<details>")
            html.append("<summary>Environment</summary>")
            html.append("<table>")
            html.append("<tr><th style='width:30%'>Parameter</th><th style='width:70%'>Value</th></tr>")

            for k, v in sorted(self.environment.items()):
                html.append(f"<tr><td>{k}</td><td>{v}</td></tr>")

            html.append("</table>")
            html.append("</details>")  # Close environment section

        # Fields Section
        if self.adata_fields:
            # Calculate statistics for the fields section
            total_fields = sum(len(fields) for fields in self.adata_fields.values())
            missing_count = len(self.missing_fields)
            overwritten_count = len(self.overwritten_fields)
            active_fields = total_fields - overwritten_count - missing_count

            # Add section header with stats
            html.append("<details>")
            html.append("<summary>Fields Created by This Run</summary>")
            html.append("<div class='field-section'>")

            # Show field statistics
            html.append("<div class='summary-box'>")
            html.append(f"<strong>Total Fields:</strong> {total_fields} &nbsp;|&nbsp; ")
            html.append(f"<strong>Present:</strong> {active_fields} &nbsp;|&nbsp; ")
            html.append(f"<strong>Missing:</strong> {missing_count} &nbsp;|&nbsp; ")
            html.append(f"<strong>Overwritten:</strong> {overwritten_count}")
            html.append("</div>")
            
            # Add fields table
            html.append("<table>")
            html.append("<tr>")
            html.append("<th style='width:40%'>Field Name</th>")
            html.append("<th style='width:10%'>Location</th>")
            html.append("<th style='width:35%'>Description</th>")
            html.append("<th style='width:15%'>Status</th>")
            html.append("</tr>")
            
            # Get all field info for formatting
            all_fields = []
            raw_data = self.get_raw_data()
            field_mapping = raw_data.get('field_mapping', {})
            
            # If field_mapping is a string, try to parse it as JSON
            if isinstance(field_mapping, str):
                try:
                    from .json_utils import from_json_string
                    field_mapping = from_json_string(field_mapping)
                except Exception as e:
                    logger.warning(f"Error parsing field_mapping as JSON: {e}")
                    field_mapping = {}
            
            # Collect all fields with their metadata
            for location, fields in self.adata_fields.items():
                for field in fields:
                    # Get field metadata
                    field_info = {
                        'name': field,
                        'location': location,
                        'type': None,
                        'description': None,
                        'overwritten': None
                    }
                    
                    # Check if field was overwritten
                    overwritten_info = next((info for info in self.overwritten_fields if
                                        info['location'] == location and info['field'] == field), None)
                    if overwritten_info:
                        field_info['overwritten'] = overwritten_info['current_run_id']

                    # Check if field is missing
                    missing_info = next((info for info in self.missing_fields if
                                        info['location'] == location and info['field'] == field), None)
                    field_info['missing'] = missing_info is not None
                    
                    # Get additional info from field_mapping
                    if field in field_mapping:
                        mapping = field_mapping[field]
                        # Handle case where mapping might be a JSON string
                        if isinstance(mapping, str):
                            try:
                                from .json_utils import from_json_string
                                mapping = from_json_string(mapping)
                            except Exception:
                                mapping = {}
                                
                        if isinstance(mapping, dict):
                            field_info['location'] = mapping.get('location', location)
                            field_info['type'] = mapping.get('type')
                            field_info['description'] = mapping.get('description')
                    
                    all_fields.append(field_info)
            
            # Group fields by location
            fields_by_location = {}
            for field_info in all_fields:
                location = field_info['location']
                if location not in fields_by_location:
                    fields_by_location[location] = []
                fields_by_location[location].append(field_info)
            
            # Sort locations and fields within locations
            for location in sorted(fields_by_location.keys()):
                fields_by_location[location].sort(key=lambda x: x['name'])
                
                # Add location header
                html.append(f"<tr><td colspan='4' class='location-header'>{location.upper()} Fields</td></tr>")
                
                # Add fields for this location
                for field_info in fields_by_location[location]:
                    name = field_info['name']
                    description = field_info['description'] or ""
                    field_type = field_info['type'] or ""
                    if field_type:
                        description = f"<small>[{field_type}]</small> {description}"
                    
                    overwritten = field_info['overwritten']
                    missing = field_info.get('missing', False)

                    row_class = "field-row"
                    status_class = "active"
                    status_text = "Present"

                    # Priority: missing > overwritten > present
                    if missing:
                        status_class = "overwritten"  # Use red styling
                        status_text = "Missing/Deleted"
                    elif overwritten:
                        status_class = "overwritten"
                        status_text = f"Overwritten by Run {overwritten}"

                    html.append(f"<tr class='{row_class}'>")
                    html.append(f"<td>{name}</td>")
                    html.append(f"<td>{location}</td>")
                    html.append(f"<td>{description}</td>")
                    html.append(f"<td class='{status_class}'>{status_text}</td>")
                    html.append("</tr>")
            
            html.append("</table>")
            html.append("</div>")  # Close field-section div
            html.append("</details>")  # Close fields section

        html.append("</div>")  # Close runinfo div
        
        return "".join(html)


class RunComparison:
    """
    Class for comparing two runs of differential analysis.
    
    Attributes
    ----------
    adata : AnnData
        AnnData object containing the run history
    run1 : RunInfo
        First run to compare
    run2 : RunInfo
        Second run to compare
    """
    
    def __init__(self, 
                adata: AnnData, 
                run_id1: int, 
                run_id2: int, 
                analysis_type: str):
        """
        Initialize a RunComparison object.
        
        Parameters
        ----------
        adata : AnnData
            AnnData object containing run history
        run_id1 : int
            First run ID to compare
        run_id2 : int
            Second run ID to compare
        analysis_type : str
            Type of analysis: 'de' for differential expression or 'da' for differential abundance
        """
        self.adata = adata
        self.analysis_type = analysis_type
        
        # Create RunInfo objects for both runs
        self.run1 = RunInfo(adata, run_id=run_id1, analysis_type=analysis_type)
        self.run2 = RunInfo(adata, run_id=run_id2, analysis_type=analysis_type)
        
        # Get summaries
        self.summary1 = self.run1.get_summary()
        self.summary2 = self.run2.get_summary()
        
        # Compare parameters
        self.param_comparison = self._compare_parameters()
        
        # Compare fields
        self.field_comparison = self._compare_fields()
    
    def _compare_parameters(self) -> Dict[str, Any]:
        """
        Compare parameters between runs.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with parameter comparison results
        """
        # Get parameters from both runs
        params1 = self.run1.params
        params2 = self.run2.params
        
        # Find common, unique, and different parameters
        common_keys = set(params1.keys()).intersection(set(params2.keys()))
        
        # Categorize parameters
        same_params = {}
        different_params = {}
        
        for key in common_keys:
            if params1[key] == params2[key]:
                same_params[key] = params1[key]
            else:
                different_params[key] = {'run1': params1[key], 'run2': params2[key]}
        
        # Find unique parameters
        only_in_run1 = {k: params1[k] for k in params1 if k not in params2}
        only_in_run2 = {k: params2[k] for k in params2 if k not in params1}
        
        return {
            'same': same_params,
            'different': different_params,
            'only_in_run1': only_in_run1,
            'only_in_run2': only_in_run2
        }
    
    def _compare_fields(self) -> Dict[str, Any]:
        """
        Compare fields between runs.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with field comparison results
        """
        # Get fields from both runs
        fields1 = self.run1.adata_fields
        fields2 = self.run2.adata_fields
        
        # Initialize comparison data structure
        comparison = {'by_location': {}}
        
        # Build sets of locations and fields
        all_locations = set(fields1.keys()).union(set(fields2.keys()))
        
        # Compare fields by location
        for location in all_locations:
            # Initialize location comparison
            comparison['by_location'][location] = {
                'same': [],
                'only_in_run1': [],
                'only_in_run2': []
            }
            
            # Get field sets for this location
            fields_set1 = set(fields1.get(location, []))
            fields_set2 = set(fields2.get(location, []))
            
            # Compute same and different fields
            comparison['by_location'][location]['same'] = sorted(list(fields_set1.intersection(fields_set2)))
            comparison['by_location'][location]['only_in_run1'] = sorted(list(fields_set1 - fields_set2))
            comparison['by_location'][location]['only_in_run2'] = sorted(list(fields_set2 - fields_set1))
        
        # Add totals
        total_same = sum(len(data['same']) for data in comparison['by_location'].values())
        total_only_in_run1 = sum(len(data['only_in_run1']) for data in comparison['by_location'].values())
        total_only_in_run2 = sum(len(data['only_in_run2']) for data in comparison['by_location'].values())
        
        comparison['totals'] = {
            'same': total_same,
            'only_in_run1': total_only_in_run1,
            'only_in_run2': total_only_in_run2
        }
        
        return comparison
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the comparison.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with comparison summary
        """
        return {
            'run1': {
                'run_id': self.run1.adjusted_run_id,
                'timestamp': self.summary1['timestamp'],
                'result_key': self.run1.params.get('result_key', 'unknown')
            },
            'run2': {
                'run_id': self.run2.adjusted_run_id,
                'timestamp': self.summary2['timestamp'],
                'result_key': self.run2.params.get('result_key', 'unknown')
            },
            'parameters': {
                'same_count': len(self.param_comparison['same']),
                'different_count': len(self.param_comparison['different']),
                'only_in_run1_count': len(self.param_comparison['only_in_run1']),
                'only_in_run2_count': len(self.param_comparison['only_in_run2'])
            },
            'fields': self.field_comparison['totals']
        }
    
    def __repr__(self) -> str:
        """
        String representation of the RunComparison object.
        
        Returns
        -------
        str
            String representation
        """
        summary = self.get_summary()
        return (f"RunComparison(run1={summary['run1']['run_id']}, "
                f"run2={summary['run2']['run_id']}, "
                f"same_fields={summary['fields']['same']}, "
                f"different_params={summary['parameters']['different_count']})")
    
    def _repr_html_(self) -> str:
        """
        HTML representation for Jupyter notebooks.
        
        Returns
        -------
        str
            HTML-formatted comparison
        """
        # CSS styles for better formatting and interactivity
        css_styles = """
        <style>
            .kompot-comparison {
                max-width: 900px;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
            }
            .kompot-comparison h3 {
                margin-bottom: 10px;
                font-weight: 600;
            }
            .kompot-comparison details {
                margin-top: 15px;
                margin-bottom: 10px;
            }
            .kompot-comparison summary {
                color: #555;
                cursor: pointer;
                font-weight: 500;
                font-size: 1.1em;
                padding: 5px 0;
                list-style: none;
                user-select: none;
            }
            .kompot-comparison summary::-webkit-details-marker {
                display: none;
            }
            .kompot-comparison summary::before {
                content: "▶ ";
                display: inline-block;
                font-size: 0.8em;
                color: #888;
                margin-right: 5px;
            }
            .kompot-comparison details[open] > summary::before {
                content: "▼ ";
            }
            .kompot-comparison table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 15px;
            }
            .kompot-comparison th {
                background-color: rgba(0, 0, 0, 0.04);
                text-align: left;
                padding: 8px;
                border: 1px solid rgba(0, 0, 0, 0.12);
                font-weight: 500;
            }
            .kompot-comparison td {
                padding: 6px 8px;
                border: 1px solid rgba(0, 0, 0, 0.12);
                vertical-align: top;
            }
            .kompot-comparison .badge {
                display: inline-block;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.85em;
                font-weight: 500;
                margin-left: 5px;
            }
            .kompot-comparison .badge-success {
                background-color: rgba(46, 125, 50, 0.12);
                color: #2e7d32;
                border: 1px solid rgba(46, 125, 50, 0.3);
            }
            .kompot-comparison .badge-warning {
                background-color: rgba(237, 108, 2, 0.12);
                color: #e65100;
                border: 1px solid rgba(237, 108, 2, 0.3);
            }
            .kompot-comparison .badge-danger {
                background-color: rgba(198, 40, 40, 0.12);
                color: #c62828;
                border: 1px solid rgba(198, 40, 40, 0.3);
            }
            .kompot-comparison .badge-info {
                background-color: rgba(21, 101, 192, 0.12);
                color: #1565c0;
                border: 1px solid rgba(21, 101, 192, 0.3);
            }
            .kompot-comparison .summary-box {
                background-color: rgba(0, 0, 0, 0.02);
                border: 1px solid rgba(0, 0, 0, 0.08);
                padding: 8px 12px;
                margin: 10px 0;
                border-radius: 4px;
            }
            .kompot-comparison .note-box {
                background-color: rgba(0, 0, 0, 0.02);
                padding: 8px;
                margin-top: 10px;
                font-size: 0.9em;
                border-radius: 4px;
            }
            .kompot-comparison .location-header {
                background-color: rgba(0, 0, 0, 0.04);
                font-weight: bold;
            }
            .kompot-comparison .location-header-shared {
                background-color: rgba(33, 150, 243, 0.08);
                font-weight: bold;
            }
            .kompot-comparison .diff-added {
                background-color: rgba(46, 125, 50, 0.08);
            }
            .kompot-comparison .diff-removed {
                background-color: rgba(198, 40, 40, 0.08);
            }
            .kompot-comparison .diff-changed {
                background-color: rgba(237, 108, 2, 0.08);
            }
            .kompot-comparison .diff-unchanged {
                background-color: rgba(0, 0, 0, 0.02);
            }
            .kompot-comparison .field-category {
                text-align: center;
                font-weight: 600;
            }
            .kompot-comparison .not-set {
                color: #888;
                font-style: italic;
            }
            .kompot-comparison .run1-only {
                color: #c62828;
            }
            .kompot-comparison .run2-only {
                color: #2e7d32;
            }
            .kompot-comparison .in-both {
                color: #1565c0;
            }

            @media (prefers-color-scheme: dark) {
                .kompot-comparison summary {
                    color: #b8b8b8;
                }
                .kompot-comparison summary::before {
                    color: #888;
                }
                .kompot-comparison th {
                    background-color: rgba(255, 255, 255, 0.06);
                    border-color: rgba(255, 255, 255, 0.15);
                }
                .kompot-comparison td {
                    border-color: rgba(255, 255, 255, 0.15);
                }
                .kompot-comparison .badge-success {
                    background-color: rgba(102, 187, 106, 0.2);
                    color: #81c784;
                    border-color: rgba(102, 187, 106, 0.4);
                }
                .kompot-comparison .badge-warning {
                    background-color: rgba(255, 167, 38, 0.2);
                    color: #ffb74d;
                    border-color: rgba(255, 167, 38, 0.4);
                }
                .kompot-comparison .badge-danger {
                    background-color: rgba(239, 83, 80, 0.2);
                    color: #e57373;
                    border-color: rgba(239, 83, 80, 0.4);
                }
                .kompot-comparison .badge-info {
                    background-color: rgba(100, 181, 246, 0.2);
                    color: #64b5f6;
                    border-color: rgba(100, 181, 246, 0.4);
                }
                .kompot-comparison .summary-box {
                    background-color: rgba(255, 255, 255, 0.03);
                    border-color: rgba(255, 255, 255, 0.12);
                }
                .kompot-comparison .note-box {
                    background-color: rgba(255, 255, 255, 0.03);
                }
                .kompot-comparison .location-header {
                    background-color: rgba(255, 255, 255, 0.08);
                }
                .kompot-comparison .location-header-shared {
                    background-color: rgba(100, 181, 246, 0.12);
                }
                .kompot-comparison .diff-added {
                    background-color: rgba(102, 187, 106, 0.12);
                }
                .kompot-comparison .diff-removed {
                    background-color: rgba(239, 83, 80, 0.12);
                }
                .kompot-comparison .diff-changed {
                    background-color: rgba(255, 167, 38, 0.12);
                }
                .kompot-comparison .diff-unchanged {
                    background-color: rgba(255, 255, 255, 0.03);
                }
                .kompot-comparison .not-set {
                    color: #888;
                }
                .kompot-comparison .run1-only {
                    color: #ef5350;
                }
                .kompot-comparison .run2-only {
                    color: #66bb6a;
                }
                .kompot-comparison .in-both {
                    color: #64b5f6;
                }
            }
        </style>
        """
        
        # Calculate statistics for display
        total_params = len(self.param_comparison['same']) + len(self.param_comparison['different']) + \
                      len(self.param_comparison['only_in_run1']) + len(self.param_comparison['only_in_run2'])
        different_params = len(self.param_comparison['different']) + len(self.param_comparison['only_in_run1']) + \
                          len(self.param_comparison['only_in_run2'])
        same_params = len(self.param_comparison['same'])
        
        # Build HTML
        html = [
            css_styles,
            "<div class='kompot-comparison'>",
            f"<h3>Comparison of Run {self.run1.adjusted_run_id} and Run {self.run2.adjusted_run_id}</h3>",

            # Summary Section
            "<details>",
            "<summary>Summary</summary>",
            "<div class='summary-box'>",
            f"<strong>Analysis Type:</strong> {self.analysis_type.upper()} &nbsp;|&nbsp; ",
            f"<strong>Run {self.run1.adjusted_run_id}:</strong> {self.summary1.get('result_key', 'unknown')} ({self.summary1.get('timestamp', '')}) &nbsp;|&nbsp; ",
            f"<strong>Run {self.run2.adjusted_run_id}:</strong> {self.summary2.get('result_key', 'unknown')} ({self.summary2.get('timestamp', '')})",
            "</div>",
            
            # Badges for differences
            "<div style='margin: 10px 0;'>",
        ]
        
        # Parameter differences badge
        param_badge_class = "badge-success" if different_params == 0 else "badge-warning"
        html.append(f"<span class='badge {param_badge_class}'>Parameters: {different_params} different, {same_params} same</span>")
        
        # Field differences badge
        field_totals = self.field_comparison['totals']
        total_fields = field_totals['same'] + field_totals['only_in_run1'] + field_totals['only_in_run2']
        different_fields = field_totals['only_in_run1'] + field_totals['only_in_run2']
        
        field_badge_class = "badge-success" if different_fields == 0 else "badge-warning"
        html.append(f"<span class='badge {field_badge_class}'>Fields: {different_fields} different, {field_totals['same']} same</span>")
        
        html.append("</div>")  # Close badges
        
        # Run comparison table
        html.append("<table>")
        html.append("<tr><th rowspan='2'>Aspect</th><th colspan='2'>Run Details</th></tr>")
        html.append(f"<tr><th>Run {self.run1.adjusted_run_id}</th><th>Run {self.run2.adjusted_run_id}</th></tr>")
        
        # Key parameters to compare
        for aspect in ['conditions', 'result_key', 'uses_sample_variance', 'timestamp']:
            html.append("<tr>")
            html.append(f"<td>{aspect}</td>")
            
            val1 = self.run1.params.get(aspect, self.summary1.get(aspect, '-'))
            val2 = self.run2.params.get(aspect, self.summary2.get(aspect, '-'))
            
            row_class = "" if val1 == val2 else "diff-changed"
            html.append(f"<td class='{row_class}'>{val1}</td>")
            html.append(f"<td class='{row_class}'>{val2}</td>")
            html.append("</tr>")
        
        # Field counts
        html.append("<tr>")
        html.append("<td>Field Count</td>")
        field_count1 = sum(len(fields) for fields in self.run1.adata_fields.values())
        field_count2 = sum(len(fields) for fields in self.run2.adata_fields.values())
        row_class = "" if field_count1 == field_count2 else "diff-changed"
        html.append(f"<td class='{row_class}'>{field_count1}</td>")
        html.append(f"<td class='{row_class}'>{field_count2}</td>")
        html.append("</tr>")
        
        html.append("</table>")
        html.append("</details>")  # Close summary section

        # Parameter Differences Section (expanded by default)
        html.append("<details open>")
        html.append("<summary>Parameter Differences</summary>")
        
        # Prepare badges to highlight important parameter differences
        param_badges = []
        if 'condition1' in self.param_comparison['different'] or 'condition2' in self.param_comparison['different']:
            param_badges.append("<span class='badge badge-danger'>Different Conditions</span>")
        if 'groupby' in self.param_comparison['different']:
            param_badges.append("<span class='badge badge-danger'>Different Group Variable</span>")
        if 'obsm_key' in self.param_comparison['different']:
            param_badges.append("<span class='badge badge-danger'>Different Embedding</span>")
        if 'layer' in self.param_comparison['different']:
            param_badges.append("<span class='badge badge-danger'>Different Layer</span>")
        if 'use_sample_variance' in self.param_comparison['different']:
            param_badges.append("<span class='badge badge-warning'>Different Variance Method</span>")
        
        # Add stats for parameters with badges if relevant
        html.append("<div class='summary-box'>")
        html.append(f"<strong>Total Parameters:</strong> {total_params} &nbsp;|&nbsp; ")
        html.append(f"<strong>Different:</strong> {len(self.param_comparison['different'])} &nbsp;|&nbsp; ")
        html.append(f"<strong>Only in Run {self.run1.adjusted_run_id}:</strong> {len(self.param_comparison['only_in_run1'])} &nbsp;|&nbsp; ")
        html.append(f"<strong>Only in Run {self.run2.adjusted_run_id}:</strong> {len(self.param_comparison['only_in_run2'])}")
        
        # Add parameter difference badges if any
        if param_badges:
            html.append("<div style='margin-top:8px'>")
            html.append(" ".join(param_badges))
            html.append("</div>")
            
        html.append("</div>")
        
        # Highlight key comparison parameters at the top
        key_params = ['condition1', 'condition2', 'groupby', 'obsm_key', 'layer', 'use_sample_variance', 'result_key']
        key_param_diffs = []
        for param in key_params:
            if param in self.param_comparison['different']:
                key_param_diffs.append(param)
        
        if key_param_diffs:
            # Create a special table for key parameters
            html.append("<h5 style='margin-top:15px; color:#d32f2f;'>Key Parameter Differences</h5>")
            html.append("<table>")
            html.append("<tr>")
            html.append(f"<th style='width:30%'>Parameter</th>")
            html.append(f"<th style='width:35%'>Run {self.run1.adjusted_run_id}</th>")
            html.append(f"<th style='width:35%'>Run {self.run2.adjusted_run_id}</th>")
            html.append("</tr>")
            
            for param in key_param_diffs:
                values = self.param_comparison['different'][param]
                html.append("<tr class='diff-changed'>")
                html.append(f"<td><strong>{param}</strong></td>")
                html.append(f"<td>{values['run1']}</td>")
                html.append(f"<td>{values['run2']}</td>")
                html.append("</tr>")
                
            html.append("</table>")
        
        # Create parameter difference table for all other differences
        if self.param_comparison['different'] or self.param_comparison['only_in_run1'] or self.param_comparison['only_in_run2']:
            
            html.append("<h5 style='margin-top:15px;'>All Parameter Differences</h5>")
            html.append("<table>")
            html.append("<tr>")
            html.append(f"<th style='width:30%'>Parameter</th>")
            html.append(f"<th style='width:35%'>Run {self.run1.adjusted_run_id}</th>")
            html.append(f"<th style='width:35%'>Run {self.run2.adjusted_run_id}</th>")
            html.append("</tr>")
            
            # Add different parameters first
            if self.param_comparison['different']:
                html.append("<tr><td colspan='3' class='field-category diff-changed'>Different Parameters</td></tr>")
                for param, values in sorted(self.param_comparison['different'].items()):
                    # Skip key parameters already shown above
                    if key_param_diffs and param in key_param_diffs:
                        continue
                        
                    html.append("<tr class='diff-changed'>")
                    html.append(f"<td>{param}</td>")
                    html.append(f"<td>{values['run1']}</td>")
                    html.append(f"<td>{values['run2']}</td>")
                    html.append("</tr>")
            
            # Add parameters only in run1
            if self.param_comparison['only_in_run1']:
                html.append(f"<tr><td colspan='3' class='field-category diff-removed'>Only in Run {self.run1.adjusted_run_id}</td></tr>")
                for param, value in sorted(self.param_comparison['only_in_run1'].items()):
                    html.append("<tr class='diff-removed'>")
                    html.append(f"<td>{param}</td>")
                    html.append(f"<td>{value}</td>")
                    html.append(f"<td class='not-set'>not set</td>")
                    html.append("</tr>")
            
            # Add parameters only in run2
            if self.param_comparison['only_in_run2']:
                html.append(f"<tr><td colspan='3' class='field-category diff-added'>Only in Run {self.run2.adjusted_run_id}</td></tr>")
                for param, value in sorted(self.param_comparison['only_in_run2'].items()):
                    html.append("<tr class='diff-added'>")
                    html.append(f"<td>{param}</td>")
                    html.append(f"<td class='not-set'>not set</td>")
                    html.append(f"<td>{value}</td>")
                    html.append("</tr>")
            
            # Add common parameters (if not too many)
            if len(self.param_comparison['same']) <= 5:
                html.append("<tr><td colspan='3' class='field-category diff-unchanged'>Same Parameters</td></tr>")
                for param, value in sorted(self.param_comparison['same'].items()):
                    html.append("<tr class='diff-unchanged'>")
                    html.append(f"<td>{param}</td>")
                    html.append(f"<td>{value}</td>")
                    html.append(f"<td>{value}</td>")
                    html.append("</tr>")
            else:
                # Just show count if many common parameters
                html.append("<tr><td colspan='3' class='field-category diff-unchanged'>")
                html.append(f"{len(self.param_comparison['same'])} parameters are the same in both runs")
                html.append("</td></tr>")
                
            html.append("</table>")
        else:
            html.append("<p><em>All parameters are identical between runs</em></p>")
        
        html.append("</details>")  # Close parameters section

        # Field Differences Section
        html.append("<details>")
        html.append("<summary>Field Differences</summary>")

        # Show field difference statistics
        html.append("<div class='summary-box'>")
        html.append(f"<strong>Fields in both runs:</strong> {self.field_comparison['totals']['same']} &nbsp;|&nbsp; ")
        html.append(f"<strong>Only in Run {self.run1.adjusted_run_id}:</strong> {self.field_comparison['totals']['only_in_run1']} &nbsp;|&nbsp; ")
        html.append(f"<strong>Only in Run {self.run2.adjusted_run_id}:</strong> {self.field_comparison['totals']['only_in_run2']}")
        html.append("</div>")
        
        # Organize fields by location for better display
        fields_by_location = {}
        for location, diffs in self.field_comparison['by_location'].items():
            if location not in fields_by_location:
                fields_by_location[location] = {'same': [], 'only_in_run1': [], 'only_in_run2': []}
            
            # Add fields to appropriate category
            for category in ['same', 'only_in_run1', 'only_in_run2']:
                for field in diffs.get(category, []):
                    fields_by_location[location][category].append(field)
        
        # Get field ownership information from the AnnData object's tracking
        storage_key = f"kompot_{self.analysis_type}"
        field_ownership = {}
        
        if (storage_key in self.adata.uns and 'anndata_fields' in self.adata.uns[storage_key]):
            from .json_utils import from_json_string
            tracking = self.adata.uns[storage_key]['anndata_fields']
            
            # If tracking is a JSON string, deserialize it
            if isinstance(tracking, str):
                tracking = from_json_string(tracking)
                
            # Collect ownership information for each field by location
            for location in tracking:
                if location not in field_ownership:
                    field_ownership[location] = {}
                    
                location_tracking = tracking[location]
                # If location_tracking is a JSON string, deserialize it
                if isinstance(location_tracking, str):
                    location_tracking = from_json_string(location_tracking)
                    
                for field, run_id in location_tracking.items():
                    field_ownership[location][field] = run_id
        
        # Create field difference table
        html.append("<table>")
        html.append("<tr>")
        html.append("<th style='width:40%'>Field Name</th>")
        html.append("<th style='width:10%'>Location</th>")
        html.append("<th style='width:25%'>Status</th>")
        html.append("<th style='width:25%'>Last Modified By</th>")
        html.append("</tr>")
        
        # First display fields that appear in both runs (overlapping fields)
        has_overlapping_fields = False
        for location in sorted(fields_by_location.keys()):
            loc_data = fields_by_location[location]
            
            if loc_data['same']:
                has_overlapping_fields = True
                
                # Display location header
                html.append(f"<tr><td colspan='4' class='location-header-shared'>{location.upper()} Shared Fields</td></tr>")
                
                # Display fields in both runs with ownership information
                for field in sorted(loc_data['same']):
                    owner_run_id = None
                    owner_class = ""
                    owner_text = "Unknown"
                    status_text = "Defined in both runs"
                    
                    # Check ownership from field_tracking
                    if location in field_ownership and field in field_ownership[location]:
                        owner_run_id = field_ownership[location][field]
                        
                        if owner_run_id == self.run1.adjusted_run_id:
                            owner_class = "run1-only"
                            owner_text = f"Run {self.run1.adjusted_run_id}"
                            status_text = f"Current value from Run {self.run1.adjusted_run_id}"
                        elif owner_run_id == self.run2.adjusted_run_id:
                            owner_class = "run2-only"
                            owner_text = f"Run {self.run2.adjusted_run_id}"
                            status_text = f"Current value from Run {self.run2.adjusted_run_id}"
                        else:
                            owner_class = "in-both"
                            owner_text = f"Run {owner_run_id} (Other)"
                            status_text = f"Current value from Run {owner_run_id}"
                    
                    html.append("<tr class='diff-unchanged'>")
                    html.append(f"<td>{field}</td>")
                    html.append(f"<td>{location}</td>")
                    html.append(f"<td>{status_text}</td>")
                    html.append(f"<td class='{owner_class}'>{owner_text}</td>")
                    html.append("</tr>")
        
        # Then display differing fields
        has_diff_fields = False
        for location in sorted(fields_by_location.keys()):
            loc_data = fields_by_location[location]
            
            if loc_data['only_in_run1'] or loc_data['only_in_run2']:
                has_diff_fields = True
                
                # Display location header
                html.append(f"<tr><td colspan='4' class='location-header'>{location.upper()} Different Fields</td></tr>")
                
                # Display fields only in run1
                for field in sorted(loc_data['only_in_run1']):
                    html.append("<tr class='diff-removed'>")
                    html.append(f"<td>{field}</td>")
                    html.append(f"<td>{location}</td>")
                    html.append(f"<td>Only in Run {self.run1.adjusted_run_id}</td>")
                    html.append(f"<td class='run1-only'>Run {self.run1.adjusted_run_id}</td>")
                    html.append("</tr>")
                
                # Display fields only in run2
                for field in sorted(loc_data['only_in_run2']):
                    html.append("<tr class='diff-added'>")
                    html.append(f"<td>{field}</td>")
                    html.append(f"<td>{location}</td>")
                    html.append(f"<td>Only in Run {self.run2.adjusted_run_id}</td>")
                    html.append(f"<td class='run2-only'>Run {self.run2.adjusted_run_id}</td>")
                    html.append("</tr>")
        
        html.append("</table>")
        
        if not has_overlapping_fields and not has_diff_fields:
            html.append("<p><em>No fields found to compare.</em></p>")
            
        # Add explanation text instead of legend
        if has_overlapping_fields:
            html.append("<div class='note-box'>")
            html.append("<strong>Note on shared fields:</strong> When both runs define the same field, the last run to write to the field ")
            html.append("overwrites the previous value. The 'Last Modified By' column shows which run's value is currently stored.")
            html.append("</div>")

        html.append("</details>")  # Close fields section

        html.append("</div>")  # Close comparison div
        
        return "".join(html)
    