# neuro_copilot/core/error_recovery.py

"""
Error recovery and interactive clarification system.

This module provides utilities for:
1. Detecting recoverable errors (e.g., missing fields)
2. Generating clarification questions for users
3. Applying user clarifications to resume execution
"""

from typing import Dict, Any, List, Optional, Tuple
import re


class RecoverableError:
    """
    Represents an error that can be recovered from through user clarification.
    """
    def __init__(
        self,
        error_type: str,
        error_message: str,
        suggested_questions: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
    ):
        self.error_type = error_type
        self.error_message = error_message
        self.suggested_questions = suggested_questions
        self.context = context or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type,
            "error_message": self.error_message,
            "suggested_questions": self.suggested_questions,
            "context": self.context,
            "recoverable": True,
        }


def analyze_error_for_recovery(
    error: Exception,
    error_message: str,
    step_params: Dict[str, Any],
    field_catalog: Optional[Dict[str, Any]] = None,
    answered_fields: Optional[set] = None,
) -> Optional[RecoverableError]:
    """
    Analyze an error to determine if it's recoverable and what questions to ask.
    
    Args:
        error: The exception that occurred
        error_message: Error message string
        step_params: Parameters of the failed step
        field_catalog: Dataset field catalog for context
        answered_fields: Set of field names that have already been clarified by the user
    
    Returns:
        RecoverableError if the error is recoverable, None otherwise
    """
    error_type = type(error).__name__
    answered_fields = answered_fields or set()
    
    # Pattern 1: Missing field errors
    if error_type == "RuntimeError" and ("missing" in error_message.lower() or "not found" in error_message.lower()):
        missing_fields = extract_missing_fields_from_error(error_message, step_params)
        
        # Filter out already answered fields
        missing_fields = [f for f in missing_fields if f not in answered_fields]
        
        if missing_fields:
            questions = generate_missing_field_questions(
                missing_fields,
                field_catalog=field_catalog,
                step_params=step_params,
            )
            
            return RecoverableError(
                error_type="missing_fields",
                error_message=error_message,
                suggested_questions=questions,
                context={
                    "missing_fields": missing_fields,
                    "step_params": step_params,
                },
            )
        else:
            # All missing fields were already answered - not a recoverable error anymore
            # This means the clarifications weren't properly applied
            return None
    
    # Pattern 2: Field shape mismatch or type errors
    if "shape" in error_message.lower() or "dimension" in error_message.lower():
        # Could ask about expected data format
        questions = [{
            "field": "data_format",
            "question": f"There was a data shape/dimension issue: {error_message[:200]}. Could you clarify the expected format of your data?"
        }]
        
        return RecoverableError(
            error_type="shape_mismatch",
            error_message=error_message,
            suggested_questions=questions,
            context={"step_params": step_params},
        )
    
    # Pattern 3: Too few samples after filtering
    if "too few" in error_message.lower() and "sample" in error_message.lower():
        # This might be due to field mismatch
        missing_fields = extract_missing_fields_from_error(error_message, step_params)
        
        if missing_fields:
            questions = generate_missing_field_questions(
                missing_fields,
                field_catalog=field_catalog,
                step_params=step_params,
            )
            questions[0]["question"] = (
                f"The analysis couldn't proceed because required field(s) were missing, "
                f"resulting in too few usable samples. " + questions[0]["question"]
            )
            
            return RecoverableError(
                error_type="insufficient_samples_due_to_missing_fields",
                error_message=error_message,
                suggested_questions=questions,
                context={
                    "missing_fields": missing_fields,
                    "step_params": step_params,
                },
            )
    
    # Not a recoverable error
    return None


def extract_missing_fields_from_error(
    error_message: str,
    step_params: Dict[str, Any],
) -> List[str]:
    """
    Extract field names that are missing from the error message.
    """
    missing_fields = []
    
    # Pattern 1: "Missing predictor fields: - 'field_name'"
    pattern1 = r"['-]'([^']+)':\s*missing"
    matches = re.findall(pattern1, error_message, re.IGNORECASE)
    missing_fields.extend(matches)
    
    # Pattern 2: "'field_name': missing in N trials"
    pattern2 = r"'([^']+)':\s*missing\s+in\s+\d+"
    matches = re.findall(pattern2, error_message)
    missing_fields.extend(matches)
    
    # Pattern 3: Check step_params for field references mentioned in the error
    error_lower = error_message.lower()
    for param_name in ["predictors", "target", "condition_field", "response_field"]:
        if param_name in step_params:
            param_value = step_params[param_name]
            if isinstance(param_value, str):
                if param_value.lower() in error_lower:
                    missing_fields.append(param_value)
            elif isinstance(param_value, list):
                for item in param_value:
                    if isinstance(item, str) and item.lower() in error_lower:
                        missing_fields.append(item)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_missing = []
    for field in missing_fields:
        if field not in seen:
            seen.add(field)
            unique_missing.append(field)
    
    return unique_missing


def generate_missing_field_questions(
    missing_fields: List[str],
    field_catalog: Optional[Dict[str, Any]] = None,
    step_params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """
    Generate clarification questions for missing fields.
    """
    questions = []
    
    # Check if there are matrix structures that might contain the field
    matrix_structures = []
    if field_catalog:
        matrix_structures = field_catalog.get("matrix_structures", [])
    
    for field_name in missing_fields:
        question_text = f"I couldn't find a field named '{field_name}' in the dataset. "
        
        # Check if there's a matrix structure that might represent this field
        if matrix_structures:
            # For common categorical fields like cue_location, suggest column index
            field_lower = field_name.lower()
            if any(keyword in field_lower for keyword in ["location", "position", "cue", "stimulus", "direction", "condition"]):
                # Find the most likely matrix
                for ms in matrix_structures:
                    n_cols = ms.get("n_cols", 0)
                    if n_cols >= 4:  # Reasonable number of categories
                        var_name = ms.get("variable_name", "unknown")
                        col_range = ms.get("column_index_range", "")
                        question_text += (
                            f"However, I noticed that the dataset has a matrix structure "
                            f"(variable '{var_name}' with {n_cols} columns, indices {col_range}). "
                            f"Is '{field_name}' represented by the **column index** in this matrix?"
                        )
                        break
                else:
                    question_text += (
                        f"Could you clarify: \n"
                        f"1. Is this field stored under a different name? If so, what is it?\n"
                        f"2. Is this information encoded in the data structure (e.g., row/column indices)?\n"
                        f"3. Can this field be computed from other available fields?"
                    )
            else:
                question_text += (
                    f"Could you clarify: \n"
                    f"1. Is this field stored under a different name? If so, what is it?\n"
                    f"2. Can this field be computed from other available fields?"
                )
        else:
            question_text += (
                f"Could you please clarify: \n"
                f"1. Is this field stored under a different name in the dataset?\n"
                f"2. Can this field be computed or derived from other available fields?"
            )
        
        questions.append({
            "field": field_name,
            "question": question_text,
        })
    
    return questions


def apply_user_clarifications(
    plan: Dict[str, Any],
    clarifications: str,
    recoverable_error: Optional[RecoverableError] = None,
) -> Dict[str, Any]:
    """
    Apply user clarifications to update the plan.
    
    This is a placeholder for now - the actual logic will be handled
    by re-running the reasoning agent with enhanced context.
    
    Args:
        plan: Original plan that failed
        clarifications: User's clarification text
        recoverable_error: The recoverable error context
    
    Returns:
        Updated plan (or original plan to be re-processed by reasoning agent)
    """
    # For now, we just return the original plan
    # The actual update will happen when reasoning agent re-runs with enhanced context
    return plan

