"""
Schema Helper Functions for Generated Code

These functions are called by Code Generator's generated code to:
1. Extract column names from schema by semantic role
2. Handle schema-conditional logic
3. Validate data requirements

This ensures NO HARDCODED field names in generated code.
"""

from typing import Dict, List, Optional, Any
from neuro_copilot.core.DA.schema_parser import ParsedSchema, ColumnRole


class SchemaColumnMapper:
    """Helper to map semantic roles to actual column names"""

    def __init__(self, schema: ParsedSchema):
        self.schema = schema

    def get_column_by_role(
        self,
        role: str,
        file_name: Optional[str] = None,
        required: bool = True
    ) -> Optional[str]:
        """
        Get actual column name by semantic role.

        Args:
            role: Semantic role (e.g., 'spike_time', 'unit_id')
            file_name: Optional file name to narrow search
            required: If True, raise error if not found

        Returns:
            Column name, or None if not found and not required
        """
        # Search through schema files
        for fname, file_schema in self.schema.files.items():
            if file_name and fname != file_name:
                continue

            for col_mapping in file_schema.columns:
                if col_mapping.role.value == role:
                    return col_mapping.column_name

        if required:
            raise ValueError(
                f"Schema does not contain column with role '{role}'"
                + (f" in file '{file_name}'" if file_name else "")
            )
        return None

    def has_role(self, role: str, file_name: Optional[str] = None) -> bool:
        """Check if schema contains a column with given role"""
        try:
            result = self.get_column_by_role(role, file_name, required=False)
            return result is not None
        except:
            return False

    def get_column_metadata(self, column_name: str, file_name: Optional[str] = None) -> Dict[str, Any]:
        """Get metadata for a specific column"""
        for fname, file_schema in self.schema.files.items():
            if file_name and fname != file_name:
                continue

            for col_mapping in file_schema.columns:
                if col_mapping.column_name == column_name:
                    return {
                        'role': col_mapping.role.value,
                        'is_relative_time': col_mapping.is_relative_time,
                        'reference_column': col_mapping.reference_column,
                        'data_type': col_mapping.data_type,
                        'description': col_mapping.description,
                    }

        return {}

    def get_alignment_strategy(self) -> Dict[str, Any]:
        """
        Infer alignment strategy from schema for trial-based spike binning.

        Returns dict with:
        - event_col: Column to align to (STIMULUS_ONSET, RESPONSE_TIME, etc.)
        - is_relative: Whether event time is relative to trial start
        - reference_col: If relative, what it's relative to
        """
        # Priority order for alignment events
        alignment_roles = [
            ColumnRole.STIMULUS_ONSET,
            ColumnRole.RESPONSE_TIME,
            ColumnRole.EVENT_TIME,
            ColumnRole.TRIAL_START,
        ]

        for role in alignment_roles:
            col_name = self.get_column_by_role(role.value, required=False)
            if col_name:
                metadata = self.get_column_metadata(col_name)
                return {
                    'event_col': col_name,
                    'event_role': role.value,
                    'is_relative': metadata.get('is_relative_time', False),
                    'reference_col': metadata.get('reference_column'),
                }

        # Fallback
        return {
            'event_col': None,
            'event_role': None,
            'is_relative': False,
            'reference_col': None,
        }

    def get_label_encoding_info(self, file_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about label column for classification"""
        # Try to find label column
        label_roles = [
            ColumnRole.TRIAL_TYPE,
            ColumnRole.RESPONSE,
            ColumnRole.OUTCOME,
            ColumnRole.LABEL,
        ]

        for role in label_roles:
            col_name = self.get_column_by_role(role.value, file_name, required=False)
            if col_name:
                metadata = self.get_column_metadata(col_name, file_name)
                return {
                    'label_col': col_name,
                    'label_role': role.value,
                    'data_type': metadata.get('data_type', 'categorical'),
                    'description': metadata.get('description', ''),
                }

        return {'label_col': None}


def validate_ndt3_requirements(schema: ParsedSchema) -> Dict[str, Any]:
    """
    Validate that schema contains necessary information for NDT3.

    Returns:
        dict with:
            - is_compatible: bool
            - missing_roles: list of missing required roles
            - warnings: list of warnings
    """
    mapper = SchemaColumnMapper(schema)

    # Required roles
    required_roles = [
        ColumnRole.SPIKE_TIME,
        ColumnRole.UNIT_ID,
        ColumnRole.TRIAL_ID,
    ]

    missing_roles = []
    for role in required_roles:
        if not mapper.has_role(role.value):
            missing_roles.append(role.value)

    # Warnings
    warnings = []

    # Check for trial structure
    if not (mapper.has_role(ColumnRole.TRIAL_START.value) or
            mapper.has_role(ColumnRole.TRIAL_END.value)):
        warnings.append("No trial start/end times found. May need manual time windowing.")

    # Check for alignment events
    if not (mapper.has_role(ColumnRole.STIMULUS_ONSET.value) or
            mapper.has_role(ColumnRole.RESPONSE_TIME.value) or
            mapper.has_role(ColumnRole.EVENT_TIME.value)):
        warnings.append("No alignment event found. Will align to trial start.")

    # Check for labels
    if not (mapper.has_role(ColumnRole.TRIAL_TYPE.value) or
            mapper.has_role(ColumnRole.LABEL.value)):
        warnings.append("No label column found. Can only do feature extraction, not classification.")

    return {
        'is_compatible': len(missing_roles) == 0,
        'missing_roles': missing_roles,
        'warnings': warnings,
        'recommended': len(missing_roles) == 0 and len(warnings) < 2,
    }


# Example usage in generated code:
"""
from neuro_copilot.core.models.schema_helpers import SchemaColumnMapper

# Initialize mapper
mapper = SchemaColumnMapper(parsed_schema)

# Get column names dynamically
spike_time_col = mapper.get_column_by_role('spike_time')
unit_id_col = mapper.get_column_by_role('unit_id')
trial_id_col = mapper.get_column_by_role('trial_id')

# Get alignment strategy
alignment = mapper.get_alignment_strategy()
event_col = alignment['event_col']
is_relative = alignment['is_relative']

# Use in binning logic
for idx, trial in trial_df.iterrows():
    if is_relative:
        ref_col = mapper.get_column_by_role('trial_start')
        align_time = trial[ref_col] + trial[event_col]
    else:
        align_time = trial[event_col]

    # ... rest of binning logic
"""
