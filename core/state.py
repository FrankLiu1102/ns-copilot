# neuro_copilot/state.py

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import pandas as pd


@dataclass
class DatasetInfo:
    path: str
    n_rows: int
    n_cols: int
    dtypes: Dict[str, str]


@dataclass
class UserDataState:
    raw_data: Optional[pd.DataFrame] = None
    dataset_info: Optional[DatasetInfo] = None


@dataclass
class NeuroGlobalState:
    """NS-Copilot global runtime state (extensible)"""
    user_data: UserDataState = field(default_factory=UserDataState)
    extra: Dict[str, Any] = field(default_factory=dict)
    
    # NEW: User clarifications for field mappings and data structure
    user_clarifications: Dict[str, str] = field(default_factory=dict)
    # Format: {"field_name": "clarification_description"}
    # Example: {"cue_location": "column_index_0_7_in_data_group", "RT": "reaction_time_in_seconds"}