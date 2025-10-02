"""
Common Imports Module

Centralizes frequently used standard library imports to reduce duplication
across the codebase. Import from this module instead of importing individually.
"""

# Standard library imports (most frequently used across 195+ files)
import os
import sys
import time
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict

# Math and data handling
import numpy as np

# Re-export commonly used items
__all__ = [
    'os', 'sys', 'time', 'json', 'logging', 'tempfile', 'Path',
    'Dict', 'List', 'Optional', 'Tuple', 'Union', 'Any',
    'dataclass', 'asdict', 'np'
]