"""
Code Executor for NS-Copilot

This module executes LLM-generated Python code in a controlled environment.

Features:
- Sandboxed execution with restricted globals
- Pre-loaded context (raw data, tools, common libraries)
- Error capture and formatted traceback
- Execution timeout (optional)
- Result extraction

Security considerations:
- Restricted built-in functions
- No file system access beyond outputs directory
- No network access
- No process spawning
"""

from __future__ import annotations

import sys
import traceback
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd

# Import generic tools for execution context
from neuro_copilot.core.tools import generic_tools

# Import schema helpers for NDT3 support
from neuro_copilot.core.models.schema_helpers import SchemaColumnMapper


# =============================================================================
# Import stripping
# =============================================================================

import re as _re

_IMPORT_RE = _re.compile(
    r'^(\s*)'            # leading whitespace
    r'(?:import\s+\S|from\s+\S)',  # import ... or from ...
)


def _strip_imports(code: str) -> str:
    """Remove import/from-import lines from LLM-generated code.

    All required libraries are pre-loaded in the execution sandbox, so
    import statements are always redundant.  Stripping them prevents
    ``KeyError('__import__')`` from the restricted builtins while keeping
    the rest of the code intact.

    - Handles ``import X``, ``from X import Y``, and multi-line variants
      using backslash continuation or parenthesised groups.
    - Replaces each removed line with ``# [auto-removed import]`` so that
      line numbers in tracebacks still match the original code.
    """
    lines = code.split('\n')
    cleaned: list[str] = []
    in_multiline = False

    for line in lines:
        if in_multiline:
            # Continue consuming a multi-line import (parenthesised or
            # backslash-continued).
            cleaned.append('# [auto-removed import]')
            if line.rstrip().endswith('\\'):
                continue
            if ')' in line:
                in_multiline = False
            continue

        stripped = line.lstrip()
        if _IMPORT_RE.match(line):
            cleaned.append('# [auto-removed import]')
            # Check for multi-line continuation
            if line.rstrip().endswith('\\') or ('(' in line and ')' not in line):
                in_multiline = True
        else:
            cleaned.append(line)

    return '\n'.join(cleaned)


# =============================================================================
# Safe __import__ fallback
# =============================================================================

# Top-level packages whose submodules are allowed in the sandbox.
# _safe_import checks whether the requested module name starts with one
# of these prefixes, so *any* submodule is automatically permitted
# (e.g. sklearn.model_selection.RepeatedStratifiedKFold) without
# maintaining an ever-growing explicit list.
_ALLOWED_TOP_PACKAGES: tuple[str, ...] = (
    # Core scientific stack
    'numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn',
    # Machine learning
    'sklearn', 'xgboost', 'lightgbm',
    # Standard library (safe subset)
    'math', 'statistics', 'collections', 'itertools', 'functools',
    'json', 'copy', 'warnings', 'time', 'datetime', 're',
    'typing', 'dataclasses', 'pathlib', 'os', 'glob',
    # EEG data loading
    'mne',
)

# Short aliases that map to a real package
_ALLOWED_ALIASES: dict[str, str] = {
    'np': 'numpy',
    'pd': 'pandas',
}


def _is_module_allowed(name: str) -> bool:
    """Check if a module name is allowed by prefix-matching top-level packages."""
    if name in _ALLOWED_ALIASES:
        return True
    return any(
        name == pkg or name.startswith(pkg + '.')
        for pkg in _ALLOWED_TOP_PACKAGES
    )


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Restricted __import__ that only allows whitelisted top-level packages.

    Uses prefix matching so any submodule of an allowed package is
    automatically permitted (e.g. ``sklearn.model_selection`` allows
    ``RepeatedStratifiedKFold`` without explicit enumeration).
    """
    import sys as _sys
    import importlib

    resolved = _ALLOWED_ALIASES.get(name, name)
    if not _is_module_allowed(resolved):
        raise ImportError(
            f"Module '{name}' is not available in the sandbox. "
            f"Use pre-loaded variables (np, pd, plt, etc.) instead."
        )

    # Try to return from sys.modules first (fast path)
    if resolved in _sys.modules:
        return _sys.modules[resolved]

    try:
        return importlib.import_module(resolved)
    except Exception:
        raise ImportError(
            f"Module '{resolved}' is allowed but could not be imported."
        )


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool                    # Whether execution completed without error
    result: Any                      # The 'result' variable from executed code
    stdout: str                      # Captured stdout
    stderr: str                      # Captured stderr
    error: Optional[str]             # Error message if failed
    error_type: Optional[str]        # Error type (e.g., "ValueError")
    traceback: Optional[str]         # Full traceback if failed
    execution_time: float            # Time taken in seconds
    variables: Dict[str, Any]        # All variables after execution
    

@dataclass
class ExecutionContext:
    """Context for code execution."""
    raw_data: Dict[str, pd.DataFrame]    # Raw data (filename -> DataFrame)
    tools: Any                            # Tools module
    output_dir: Optional[Path] = None     # Directory for saving outputs
    extra_globals: Dict[str, Any] = field(default_factory=dict)  # Additional globals
    

# =============================================================================
# Safe Built-ins
# =============================================================================

# Restricted set of built-in functions for safety
SAFE_BUILTINS = {
    # Types
    'bool': bool,
    'int': int,
    'float': float,
    'str': str,
    'list': list,
    'dict': dict,
    'tuple': tuple,
    'set': set,
    'frozenset': frozenset,
    'bytes': bytes,
    'bytearray': bytearray,
    'complex': complex,
    
    # Iteration
    'range': range,
    'enumerate': enumerate,
    'zip': zip,
    'map': map,
    'filter': filter,
    'reversed': reversed,
    'sorted': sorted,
    
    # Math
    'abs': abs,
    'min': min,
    'max': max,
    'sum': sum,
    'round': round,
    'pow': pow,
    'divmod': divmod,
    
    # Sequence operations
    'len': len,
    'all': all,
    'any': any,
    'iter': iter,
    'next': next,
    'slice': slice,
    
    # Object operations
    'hasattr': hasattr,
    'getattr': getattr,
    'setattr': setattr,
    'isinstance': isinstance,
    'issubclass': issubclass,
    'type': type,
    'id': id,
    'hash': hash,
    'callable': callable,
    'repr': repr,
    'format': format,
    
    # I/O (limited)
    'print': print,
    'input': None,  # Disabled
    
    # Exceptions
    'Exception': Exception,
    'BaseException': BaseException,
    'ValueError': ValueError,
    'TypeError': TypeError,
    'KeyError': KeyError,
    'IndexError': IndexError,
    'RuntimeError': RuntimeError,
    'NameError': NameError,
    'AttributeError': AttributeError,
    'ImportError': ImportError,
    'ModuleNotFoundError': ModuleNotFoundError,
    'StopIteration': StopIteration,
    'ZeroDivisionError': ZeroDivisionError,
    'FileNotFoundError': FileNotFoundError,
    'OSError': OSError,
    'IOError': IOError,
    
    # Other safe operations
    'chr': chr,
    'ord': ord,
    'bin': bin,
    'hex': hex,
    'oct': oct,
    'ascii': ascii,

    # None, True, False
    'None': None,
    'True': True,
    'False': False,

    # Metaprogramming (needed for class definitions in generated code)
    '__build_class__': __builtins__['__build_class__'],
    '__name__': '__main__',

    # Restricted import – resolves only whitelisted modules (fallback for
    # import patterns that _strip_imports() does not catch).
    '__import__': _safe_import,

    # Namespace access (safe - only returns dictionaries of names)
    'globals': globals,
    'locals': locals,
    'vars': vars,
    'dir': dir,

    # Object/class operations (safe - for class definitions and introspection)
    'delattr': delattr,
    'property': property,
    'classmethod': classmethod,
    'staticmethod': staticmethod,
    'super': super,
    'object': object,
}

# Dangerous built-ins to explicitly block
BLOCKED_BUILTINS = {
    'eval',
    'exec',
    'compile',
    'open',
    # '__import__ is now handled by _safe_import (whitelisted modules only)
    'globals',
    'locals',
    'vars',
    'dir',
    'delattr',
    'breakpoint',
    'exit',
    'quit',
    'help',
    'memoryview',
    'object',
    'property',
    'classmethod',
    'staticmethod',
    'super',
}


# =============================================================================
# Code Executor Class
# =============================================================================

class CodeExecutor:
    """
    Execute LLM-generated Python code safely.
    
    Provides a sandboxed environment with:
    - Restricted built-ins
    - Pre-loaded libraries (numpy, pandas, scipy)
    - Access to raw data and tools
    - Captured output and errors
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        allow_file_write: bool = True,
        timeout: Optional[float] = 300.0,
    ):
        """
        Initialize code executor.

        Args:
            output_dir: Directory for saving outputs (figures, etc.)
            allow_file_write: Whether to allow file writing to output_dir
            timeout: Execution timeout in seconds (default: 300s / 5 minutes)
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "outputs" / "code_execution"
        self.allow_file_write = allow_file_write
        self.timeout = timeout

        # Shared cache persists across multiple execute() calls on the same
        # CodeExecutor instance.  Generated code can store expensive results
        # (e.g., loaded EEG data, feature matrices) here so that subsequent
        # attempts reuse them instead of reloading from disk.
        self.shared_cache: Dict[str, Any] = {}

        # Ensure output directory exists
        if self.allow_file_write:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def execute(
        self,
        code: str,
        raw_data: Dict[str, pd.DataFrame],
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Execute code in sandboxed environment.
        
        Args:
            code: Python code string to execute
            raw_data: Dict mapping filename -> DataFrame
            extra_context: Additional variables to inject
            
        Returns:
            ExecutionResult with success status, result, and any errors
        """
        # Build execution globals
        exec_globals = self._build_globals(raw_data, extra_context)

        # NOTE: We no longer strip import statements.  Instead, _safe_import
        # (set as __import__ in SAFE_BUILTINS) handles security by only
        # allowing whitelisted top-level packages.  This lets LLM-generated
        # code import any submodule of allowed packages (e.g.
        # ``from sklearn.model_selection import RepeatedStratifiedKFold``)
        # without maintaining an explicit list of every class/function.

        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = captured_stdout = StringIO()
        sys.stderr = captured_stderr = StringIO()
        
        start_time = time.perf_counter()
        success = False
        result = None
        error = None
        error_type = None
        tb = None

        try:
            if self.timeout and self.timeout > 0:
                # Execute with timeout using a thread
                import threading
                exec_exception = [None]
                def _run_code():
                    try:
                        exec(code, exec_globals)
                    except Exception as e:
                        exec_exception[0] = e

                thread = threading.Thread(target=_run_code, daemon=True)
                thread.start()
                thread.join(timeout=self.timeout)

                if thread.is_alive():
                    # Timeout — thread is still running
                    elapsed = time.perf_counter() - start_time
                    # Capture partial stdout so LLM can see where execution stalled
                    _partial = captured_stdout.getvalue()
                    _partial_tail = "\n".join(_partial.strip().splitlines()[-15:]) if _partial.strip() else "(no output before timeout)"
                    error = (
                        f"Execution timed out after {elapsed:.0f}s "
                        f"(limit: {self.timeout:.0f}s). "
                        f"Partial output before timeout:\n{_partial_tail}\n\n"
                        f"Review the partial output above to identify what was running "
                        f"when the timeout occurred, and reduce the computational cost."
                    )
                    error_type = "TimeoutError"
                    tb = f"TimeoutError: Code execution exceeded {self.timeout:.0f}s limit"
                elif exec_exception[0] is not None:
                    raise exec_exception[0]
                else:
                    result = exec_globals.get('result', None)
                    success = True
            else:
                # Execute without timeout (original behavior)
                exec(code, exec_globals)
                result = exec_globals.get('result', None)
                success = True

        except Exception as e:
            error = str(e)
            error_type = type(e).__name__
            tb = traceback.format_exc()

        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        end_time = time.perf_counter()
        
        # Collect all variables (filter out modules and functions for cleaner output)
        variables = {}
        for k, v in exec_globals.items():
            if k.startswith('_'):
                continue
            if k in ('np', 'pd', 'plt', 'sns', 'tools', 'raw_data', 'scipy'):
                continue
            if callable(v) and not isinstance(v, type):
                continue
            try:
                # Try to keep simple types
                if isinstance(v, (int, float, str, bool, list, dict, tuple, np.ndarray, pd.DataFrame)):
                    variables[k] = v
            except:
                pass
        
        return ExecutionResult(
            success=success,
            result=result,
            stdout=captured_stdout.getvalue(),
            stderr=captured_stderr.getvalue(),
            error=error,
            error_type=error_type,
            traceback=tb,
            execution_time=end_time - start_time,
            variables=variables,
        )
    
    def execute_with_retry(
        self,
        code: str,
        raw_data: Dict[str, pd.DataFrame],
        max_retries: int = 3,
        fix_code_fn: Optional[callable] = None,
    ) -> Tuple[ExecutionResult, List[ExecutionResult]]:
        """
        Execute code with automatic retry on failure.
        
        Args:
            code: Initial code to execute
            raw_data: Raw data dict
            max_retries: Maximum retry attempts
            fix_code_fn: Optional function to fix code given error
                         Signature: fix_code_fn(code, error, traceback) -> new_code
            
        Returns:
            Tuple of (final_result, list_of_all_attempts)
        """
        attempts = []
        current_code = code
        
        for attempt in range(max_retries + 1):
            result = self.execute(current_code, raw_data)
            attempts.append(result)
            
            if result.success:
                return result, attempts
            
            if attempt < max_retries and fix_code_fn:
                # Try to fix the code
                try:
                    current_code = fix_code_fn(
                        current_code, 
                        result.error, 
                        result.traceback
                    )
                except Exception as e:
                    # Fix function failed, stop retrying
                    break
        
        return attempts[-1], attempts
    
    def _build_globals(
        self,
        raw_data: Dict[str, pd.DataFrame],
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the globals dict for code execution."""
        
        # Start with safe builtins
        exec_globals = {
            '__builtins__': SAFE_BUILTINS.copy(),
        }
        
        # Add numpy and pandas
        exec_globals['np'] = np
        exec_globals['numpy'] = np
        exec_globals['pd'] = pd
        exec_globals['pandas'] = pd
        
        # Add scipy if available
        try:
            import scipy
            import scipy.stats
            import scipy.signal
            exec_globals['scipy'] = scipy
        except ImportError:
            pass
        
        # Add matplotlib if available
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            exec_globals['plt'] = plt
            exec_globals['matplotlib'] = matplotlib
        except ImportError:
            pass
        
        # Add seaborn if available
        try:
            import seaborn as sns
            exec_globals['sns'] = sns
        except ImportError:
            pass
        
        # Add sklearn components commonly needed
        try:
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, LeaveOneOut
            from sklearn.pipeline import Pipeline
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            from sklearn.metrics import (
                accuracy_score, balanced_accuracy_score, 
                f1_score, roc_auc_score, confusion_matrix,
                classification_report
            )
            exec_globals['StandardScaler'] = StandardScaler
            exec_globals['LabelEncoder'] = LabelEncoder
            exec_globals['train_test_split'] = train_test_split
            exec_globals['cross_val_score'] = cross_val_score
            exec_globals['StratifiedKFold'] = StratifiedKFold
            exec_globals['LeaveOneOut'] = LeaveOneOut
            exec_globals['Pipeline'] = Pipeline
            exec_globals['LogisticRegression'] = LogisticRegression
            exec_globals['SVC'] = SVC
            exec_globals['accuracy_score'] = accuracy_score
            exec_globals['balanced_accuracy_score'] = balanced_accuracy_score
            exec_globals['f1_score'] = f1_score
            exec_globals['roc_auc_score'] = roc_auc_score
            exec_globals['confusion_matrix'] = confusion_matrix
            exec_globals['classification_report'] = classification_report
        except ImportError:
            pass
        
        # Add os and glob for file discovery (e.g., multi-file EEG datasets)
        import os as _os
        import glob as _glob
        exec_globals['os'] = _os
        exec_globals['glob'] = _glob

        # Add mne for EEG data loading (.set, .edf, etc.)
        try:
            import mne as _mne
            exec_globals['mne'] = _mne
        except ImportError:
            pass  # mne not installed; generated code will get a clear NameError

        # Add re for regex operations (e.g., extracting subject IDs from paths)
        import re as _re_mod
        exec_globals['re'] = _re_mod

        # Add raw data
        exec_globals['raw_data'] = raw_data

        # Add tools module
        exec_globals['tools'] = generic_tools

        # Add shared cache (persists across attempts on the same executor)
        exec_globals['SHARED_CACHE'] = self.shared_cache

        # Add schema helpers for NDT3 support
        exec_globals['SchemaColumnMapper'] = SchemaColumnMapper

        # Add output directory for saving figures
        exec_globals['OUTPUT_DIR'] = str(self.output_dir)

        # Add random seed for reproducibility (injected per-run from pipeline)
        _seed = getattr(self, '_random_seed', 42)
        exec_globals['RANDOM_SEED'] = _seed
        # Also inject into tools modules so tool default seed= arguments use it
        generic_tools._RANDOM_SEED = _seed
        try:
            from neuro_copilot.core.tools import decoding
            decoding._RANDOM_SEED = _seed
        except ImportError:
            pass
        
        # Add time module (needed for elapsed-time measurement in generated code)
        exec_globals['time'] = time

        # Add helper functions
        exec_globals['save_figure'] = self._create_save_figure_fn()
        
        # Add extra context
        if extra_context:
            exec_globals.update(extra_context)
        
        return exec_globals
    
    def _create_save_figure_fn(self):
        """Create a safe figure saving function."""
        output_dir = self.output_dir
        allow_write = self.allow_file_write
        
        def save_figure(fig, filename: str, dpi: int = 150):
            """Save matplotlib figure to output directory."""
            if not allow_write:
                print(f"[Warning] File writing disabled, figure not saved: {filename}")
                return None
            
            filepath = output_dir / filename
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved: {filepath}")
            return str(filepath)
        
        return save_figure


# =============================================================================
# Convenience Functions
# =============================================================================

def execute_code(
    code: str,
    raw_data: Dict[str, pd.DataFrame],
    output_dir: Optional[str] = None,
    extra_context: Optional[Dict[str, Any]] = None,
) -> ExecutionResult:
    """
    Convenience function to execute code.
    
    Args:
        code: Python code string
        raw_data: Dict mapping filename -> DataFrame
        output_dir: Directory for outputs
        extra_context: Additional variables
        
    Returns:
        ExecutionResult
    """
    executor = CodeExecutor(output_dir=output_dir)
    return executor.execute(code, raw_data, extra_context)


def execute_generated_code(
    generated_code,  # GeneratedCode from code_generator
    raw_data: Dict[str, pd.DataFrame],
    output_dir: Optional[str] = None,
) -> ExecutionResult:
    """
    Execute GeneratedCode object.
    
    Args:
        generated_code: GeneratedCode from code_generator module
        raw_data: Raw data dict
        output_dir: Output directory
        
    Returns:
        ExecutionResult
    """
    return execute_code(
        code=generated_code.code,
        raw_data=raw_data,
        output_dir=output_dir,
    )


def format_error_for_llm(result: ExecutionResult) -> str:
    """
    Format execution error for LLM to understand and fix.
    
    Args:
        result: Failed ExecutionResult
        
    Returns:
        Formatted error string for LLM prompt
    """
    if result.success:
        return ""
    
    lines = [
        "## Execution Error",
        f"Error Type: {result.error_type}",
        f"Error Message: {result.error}",
        "",
        "## Traceback",
        result.traceback or "No traceback available",
    ]
    
    if result.stdout:
        lines.extend([
            "",
            "## Stdout (before error)",
            result.stdout,
        ])
    
    return "\n".join(lines)


def format_result_summary(result: ExecutionResult) -> str:
    """
    Format execution result for display.
    
    Args:
        result: ExecutionResult
        
    Returns:
        Formatted summary string
    """
    lines = []
    
    if result.success:
        lines.append("✅ Execution successful")
        lines.append(f"⏱️ Time: {result.execution_time:.3f}s")
        
        if result.result is not None:
            lines.append("")
            lines.append("## Result")
            if isinstance(result.result, dict):
                for k, v in result.result.items():
                    if isinstance(v, (int, float)):
                        lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
                    elif isinstance(v, list) and len(v) <= 5:
                        lines.append(f"  {k}: {v}")
                    elif isinstance(v, np.ndarray):
                        lines.append(f"  {k}: array{v.shape}")
                    else:
                        lines.append(f"  {k}: {type(v).__name__}")
            else:
                lines.append(f"  {result.result}")
    else:
        lines.append("❌ Execution failed")
        lines.append(f"Error: {result.error_type}: {result.error}")
    
    if result.stdout:
        lines.extend([
            "",
            "## Output",
            result.stdout[:1000] + ("..." if len(result.stdout) > 1000 else ""),
        ])
    
    return "\n".join(lines)
