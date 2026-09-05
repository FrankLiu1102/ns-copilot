"""
Schema Parser for NS-Copilot

This module uses LLM to parse user-provided schema descriptions and output
structured semantic mappings. This is a key component for the generic data
loading architecture - no hardcoded assumptions about data types.

Flow:
1. User provides schema in prompt (Dataset Schema: section)
2. parse_schema_from_prompt() extracts raw text
3. This module uses LLM to interpret semantics
4. Output: Structured ParsedSchema for data_transformer to use
"""

import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

from neuro_copilot.core.gpt4_client import GPT4Client


class ColumnRole(str, Enum):
    """
    Semantic roles that columns can play in neural data analysis.
    These are inferred by LLM, not hardcoded in dataloader.
    """
    # Identifiers
    UNIT_ID = "unit_id"           # Neuron/unit identifier
    TRIAL_ID = "trial_id"         # Trial identifier
    SESSION_ID = "session_id"     # Session identifier
    SUBJECT_ID = "subject_id"     # Subject/animal identifier
    
    # Timestamps
    SPIKE_TIME = "spike_time"     # Individual spike timestamps
    EVENT_TIME = "event_time"     # Event timestamps
    TRIAL_START = "trial_start"   # Trial start time
    TRIAL_END = "trial_end"       # Trial end time
    STIMULUS_ONSET = "stimulus_onset"  # Stimulus onset time
    RESPONSE_TIME = "response_time"    # Response/reaction time
    
    # Neural signals
    FIRING_RATE = "firing_rate"   # Firing rate values
    CALCIUM_SIGNAL = "calcium_signal"  # Calcium imaging signal
    LFP_SIGNAL = "lfp_signal"     # Local field potential
    EEG_SIGNAL = "eeg_signal"     # EEG signal
    FMRI_SIGNAL = "fmri_signal"   # fMRI BOLD signal
    
    # Behavioral/experimental
    TRIAL_TYPE = "trial_type"     # Trial type/condition
    STIMULUS_TYPE = "stimulus_type"  # Stimulus category
    RESPONSE = "response"         # Behavioral response
    OUTCOME = "outcome"           # Trial outcome (correct/error)
    REWARD = "reward"             # Reward value
    
    # Metadata
    QUALITY = "quality"           # Unit/data quality metric
    REGION = "region"             # Brain region
    LAYER = "layer"               # Cortical layer
    CELL_TYPE = "cell_type"       # Cell type classification
    
    # Time reference
    TIME_RELATIVE = "time_relative"   # Time relative to event
    TIME_ABSOLUTE = "time_absolute"   # Absolute time
    
    # Generic
    FEATURE = "feature"           # Generic feature column
    LABEL = "label"               # Generic label column
    OTHER = "other"               # Unclassified


@dataclass
class ColumnMapping:
    """Mapping of a single column to its semantic role."""
    column_name: str
    role: ColumnRole
    description: str
    file_name: str
    is_relative_time: bool = False  # True if time is relative to trial/event
    reference_column: Optional[str] = None  # For relative times, what it's relative to
    data_type: str = "unknown"  # inferred: "numeric", "categorical", "timestamp"
    
    
@dataclass
class FileSchema:
    """Schema for a single file."""
    file_name: str
    columns: List[ColumnMapping]
    description: str = ""
    row_semantics: str = ""  # e.g., "one row per spike" or "one row per trial"


@dataclass
class ParsedSchema:
    """
    Complete parsed schema from user description.
    This is the output of LLM schema interpretation.
    """
    files: Dict[str, FileSchema] = field(default_factory=dict)
    
    # Semantic shortcuts for common operations
    spike_time_column: Optional[ColumnMapping] = None
    unit_id_column: Optional[ColumnMapping] = None
    trial_id_column: Optional[ColumnMapping] = None
    trial_type_column: Optional[ColumnMapping] = None
    stimulus_onset_column: Optional[ColumnMapping] = None
    trial_start_column: Optional[ColumnMapping] = None
    
    # Analysis hints
    data_type: str = "unknown"  # "spike", "calcium", "lfp", "behavioral", "mixed"
    time_unit: str = "seconds"  # "seconds", "milliseconds", "samples"
    
    # Raw schema text for reference
    raw_schema_text: str = ""
    
    # Parsing metadata
    confidence: float = 1.0
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "files": {},
            "data_type": self.data_type,
            "time_unit": self.time_unit,
            "confidence": self.confidence,
            "warnings": self.warnings,
        }
        
        for fname, fschema in self.files.items():
            result["files"][fname] = {
                "file_name": fschema.file_name,
                "description": fschema.description,
                "row_semantics": fschema.row_semantics,
                "columns": [
                    {
                        "column_name": c.column_name,
                        "role": c.role.value,
                        "description": c.description,
                        "is_relative_time": c.is_relative_time,
                        "reference_column": c.reference_column,
                        "data_type": c.data_type,
                    }
                    for c in fschema.columns
                ]
            }
        
        # Add semantic shortcuts
        for attr in ["spike_time_column", "unit_id_column", "trial_id_column", 
                     "trial_type_column", "stimulus_onset_column", "trial_start_column"]:
            col = getattr(self, attr)
            if col:
                result[attr] = {
                    "column_name": col.column_name,
                    "file_name": col.file_name,
                    "role": col.role.value,
                }
        
        return result


# LLM Prompt for schema parsing
SCHEMA_PARSING_SYSTEM_PROMPT = """You are a neural data schema interpreter. Your task is to analyze user-provided dataset schema descriptions and output structured JSON describing the semantic role of each column.

You understand various neural data types:
- Spike data: neuron IDs, spike timestamps, unit quality
- Calcium imaging: ROI IDs, fluorescence traces, timestamps
- LFP/EEG: channel IDs, voltage signals, timestamps
- Behavioral: trial types, responses, reaction times, outcomes
- fMRI: voxel data, BOLD signals, timepoints

For each column, determine:
1. Semantic role (from the ColumnRole enum)
2. Whether timestamps are absolute or relative
3. For relative times, what they're relative to
4. Data type (numeric, categorical, timestamp)

Output valid JSON only, no markdown code blocks."""

SCHEMA_PARSING_USER_PROMPT_TEMPLATE = """Analyze this dataset schema and output structured JSON:

{raw_schema}

Additional context from data:
{data_summary}

Output JSON with this structure:
{{
    "data_type": "spike|calcium|lfp|behavioral|mixed",
    "time_unit": "seconds|milliseconds|samples",
    "files": {{
        "filename.csv": {{
            "row_semantics": "one row per spike|one row per trial|...",
            "columns": [
                {{
                    "column_name": "column_name",
                    "role": "spike_time|unit_id|trial_type|...",
                    "description": "brief description",
                    "is_relative_time": false,
                    "reference_column": null,
                    "data_type": "numeric|categorical|timestamp"
                }}
            ]
        }}
    }},
    "warnings": ["any ambiguities or assumptions made"]
}}

Valid roles: unit_id, trial_id, session_id, subject_id, spike_time, event_time, trial_start, trial_end, stimulus_onset, response_time, firing_rate, calcium_signal, lfp_signal, trial_type, stimulus_type, response, outcome, reward, quality, region, layer, cell_type, time_relative, time_absolute, feature, label, other

Output ONLY valid JSON:"""


class SchemaParser:
    """
    LLM-based schema parser.
    
    Takes raw schema text and data summary, uses LLM to interpret
    column semantics, and outputs structured ParsedSchema.
    """
    
    def __init__(self, llm_client: Optional[GPT4Client] = None):
        """
        Initialize schema parser.
        
        Args:
            llm_client: GPT4Client instance. If None, creates new one.
        """
        self.llm_client = llm_client
        
    def _get_client(self) -> GPT4Client:
        """Lazy initialization of LLM client."""
        if self.llm_client is None:
            self.llm_client = GPT4Client()
        return self.llm_client
    
    def parse(
        self,
        raw_schema: Dict[str, Any],
        data_summary: Optional[str] = None,
    ) -> ParsedSchema:
        """
        Parse raw schema using LLM.
        
        Args:
            raw_schema: Raw schema dict from parse_schema_from_prompt()
                Example:
                {
                    "trials.csv": {"id": "Trial ID", "type": "Trial type"},
                    "spike_times.csv": {"unit_id": "Neuron ID", "spike_time": "Timestamp"},
                    "_raw_text": "Dataset Schema:\n..."
                }
            data_summary: Optional summary of actual data (column stats, samples)
                to help LLM make better inferences
                
        Returns:
            ParsedSchema with structured column mappings
        """
        # Extract raw text
        raw_text = raw_schema.get("_raw_text", "")
        if not raw_text:
            # Reconstruct from dict
            lines = ["Dataset Schema:"]
            for fname, cols in raw_schema.items():
                if fname.startswith("_"):
                    continue
                lines.append(f"  {fname}:")
                if isinstance(cols, dict):
                    for col, desc in cols.items():
                        lines.append(f"    - {col}: {desc}")
            raw_text = "\n".join(lines)
        
        # Prepare data summary
        if data_summary is None:
            data_summary = "No additional data summary available."
        
        # Build prompt
        user_prompt = SCHEMA_PARSING_USER_PROMPT_TEMPLATE.format(
            raw_schema=raw_text,
            data_summary=data_summary,
        )
        
        # Call LLM
        client = self._get_client()
        response = client.generate(
            prompt=user_prompt,
            system_prompt=SCHEMA_PARSING_SYSTEM_PROMPT,
            temperature=0.1,  # Low temperature for structured output
            max_tokens=2000,
        )
        
        # Parse LLM response
        return self._parse_llm_response(response.content, raw_schema, raw_text)
    
    def _parse_llm_response(
        self,
        llm_output: str,
        raw_schema: Dict[str, Any],
        raw_text: str,
    ) -> ParsedSchema:
        """
        Parse LLM JSON output into ParsedSchema.
        
        Args:
            llm_output: Raw LLM response (should be JSON)
            raw_schema: Original raw schema for fallback
            raw_text: Raw schema text for reference
            
        Returns:
            ParsedSchema
        """
        # Try to extract JSON from response
        json_str = llm_output.strip()
        
        # Remove markdown code blocks if present
        if json_str.startswith("```"):
            json_str = re.sub(r'^```(?:json)?\n?', '', json_str)
            json_str = re.sub(r'\n?```$', '', json_str)
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            # Fallback: create basic schema from raw input
            return self._fallback_parse(raw_schema, raw_text, f"JSON parse error: {e}")
        
        # Build ParsedSchema from JSON
        parsed = ParsedSchema(
            raw_schema_text=raw_text,
            data_type=data.get("data_type", "unknown"),
            time_unit=data.get("time_unit", "seconds"),
            warnings=data.get("warnings", []),
        )
        
        # Parse file schemas
        files_data = data.get("files", {})
        for fname, fdata in files_data.items():
            columns = []
            for col_data in fdata.get("columns", []):
                try:
                    role = ColumnRole(col_data.get("role", "other"))
                except ValueError:
                    role = ColumnRole.OTHER
                
                col = ColumnMapping(
                    column_name=col_data.get("column_name", ""),
                    role=role,
                    description=col_data.get("description", ""),
                    file_name=fname,
                    is_relative_time=col_data.get("is_relative_time", False),
                    reference_column=col_data.get("reference_column"),
                    data_type=col_data.get("data_type", "unknown"),
                )
                columns.append(col)
                
                # Set semantic shortcuts
                if role == ColumnRole.SPIKE_TIME and parsed.spike_time_column is None:
                    parsed.spike_time_column = col
                elif role == ColumnRole.UNIT_ID and parsed.unit_id_column is None:
                    parsed.unit_id_column = col
                elif role == ColumnRole.TRIAL_ID and parsed.trial_id_column is None:
                    parsed.trial_id_column = col
                elif role == ColumnRole.TRIAL_TYPE and parsed.trial_type_column is None:
                    parsed.trial_type_column = col
                elif role == ColumnRole.STIMULUS_ONSET and parsed.stimulus_onset_column is None:
                    parsed.stimulus_onset_column = col
                elif role == ColumnRole.TRIAL_START and parsed.trial_start_column is None:
                    parsed.trial_start_column = col
            
            file_schema = FileSchema(
                file_name=fname,
                columns=columns,
                description=fdata.get("description", ""),
                row_semantics=fdata.get("row_semantics", ""),
            )
            parsed.files[fname] = file_schema
        
        return parsed
    
    def _fallback_parse(
        self,
        raw_schema: Dict[str, Any],
        raw_text: str,
        error_msg: str,
    ) -> ParsedSchema:
        """
        Fallback parsing when LLM fails.
        
        Uses simple heuristics based on column names.
        """
        parsed = ParsedSchema(
            raw_schema_text=raw_text,
            warnings=[f"LLM parsing failed, using heuristic fallback: {error_msg}"],
            confidence=0.5,
        )
        
        # Heuristic role detection based on common patterns
        role_patterns = {
            r"unit_?id|neuron_?id|cell_?id|cluster_?id": ColumnRole.UNIT_ID,
            r"trial_?id": ColumnRole.TRIAL_ID,
            r"spike_?time|spike_?times": ColumnRole.SPIKE_TIME,
            r"trial_?type|condition|stim_?type": ColumnRole.TRIAL_TYPE,
            r"start_?time|trial_?start": ColumnRole.TRIAL_START,
            r"end_?time|trial_?end": ColumnRole.TRIAL_END,
            r"pole_?in|stimulus_?onset|stim_?on": ColumnRole.STIMULUS_ONSET,
            r"response|resp|rt|reaction_?time": ColumnRole.RESPONSE_TIME,
            r"outcome|correct|error|reward": ColumnRole.OUTCOME,
            r"quality|unit_?quality": ColumnRole.QUALITY,
            r"region|area|brain_?region": ColumnRole.REGION,
        }
        
        for fname, cols in raw_schema.items():
            if fname.startswith("_"):
                continue
            if not isinstance(cols, dict):
                continue
            
            columns = []
            for col_name, description in cols.items():
                # Try to match role from column name
                role = ColumnRole.OTHER
                for pattern, matched_role in role_patterns.items():
                    if re.search(pattern, col_name, re.IGNORECASE):
                        role = matched_role
                        break
                
                col = ColumnMapping(
                    column_name=col_name,
                    role=role,
                    description=description if isinstance(description, str) else str(description),
                    file_name=fname,
                )
                columns.append(col)
                
                # Set semantic shortcuts
                if role == ColumnRole.SPIKE_TIME and parsed.spike_time_column is None:
                    parsed.spike_time_column = col
                elif role == ColumnRole.UNIT_ID and parsed.unit_id_column is None:
                    parsed.unit_id_column = col
                elif role == ColumnRole.TRIAL_TYPE and parsed.trial_type_column is None:
                    parsed.trial_type_column = col
                elif role == ColumnRole.STIMULUS_ONSET and parsed.stimulus_onset_column is None:
                    parsed.stimulus_onset_column = col
                elif role == ColumnRole.TRIAL_START and parsed.trial_start_column is None:
                    parsed.trial_start_column = col
            
            file_schema = FileSchema(
                file_name=fname,
                columns=columns,
            )
            parsed.files[fname] = file_schema
        
        return parsed


def parse_schema(
    raw_schema: Dict[str, Any],
    data_summary: Optional[str] = None,
    llm_client: Optional[GPT4Client] = None,
) -> ParsedSchema:
    """
    Convenience function to parse schema.
    
    Args:
        raw_schema: Raw schema dict from parse_schema_from_prompt()
        data_summary: Optional data summary for better inference
        llm_client: Optional GPT4Client instance
        
    Returns:
        ParsedSchema
    """
    parser = SchemaParser(llm_client=llm_client)
    return parser.parse(raw_schema, data_summary)


def build_data_summary_from_raw(raw_data: Dict[str, Any]) -> str:
    """
    Build data summary string from raw loaded data.
    
    This summary helps LLM make better inferences about column semantics.
    
    Args:
        raw_data: Raw data dict from dataloader (after _load_csv_file)
        
    Returns:
        Summary string for LLM
    """
    lines = []
    
    # Check if multi-file
    if "_files" in raw_data:
        for fname, fdata in raw_data["_files"].items():
            lines.append(f"File: {fname}")
            if "_summary_text" in fdata:
                lines.append(fdata["_summary_text"])
            lines.append("")
    elif "_summary_text" in raw_data:
        lines.append(raw_data["_summary_text"])
    elif "_column_info" in raw_data:
        # Build summary from column info
        lines.append(f"Columns: {len(raw_data['_column_info'])}")
        for col, info in raw_data["_column_info"].items():
            dtype = info.get("dtype", "unknown")
            samples = info.get("sample_values", [])[:3]
            lines.append(f"  - {col} ({dtype}): samples={samples}")
    
    return "\n".join(lines) if lines else "No data summary available."
