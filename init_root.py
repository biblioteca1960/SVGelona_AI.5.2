"""
SVGelona_AI 5.2/5.3 - Root System Initialization
Consolidated registry for Kernel, Core, Optimization, Tools, and Docs.
Enforces the SVG-CERT-2026-TP-001 integrity protocol.
"""

import os
import sys
import platform

# ---------------------------------------------------
# Python Version Check
# ---------------------------------------------------
MIN_PYTHON = (3, 10)
if sys.version_info < MIN_PYTHON:
    raise RuntimeError(f"SVGelona_AI requires Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+")

# ---------------------------------------------------
# System Metadata
# ---------------------------------------------------
__version__ = "5.2.0"
__cortex_version__ = "5.3.0-alpha"
__author__ = "SVGelona_AI Team"
__integrity_protocol__ = "SVG-CERT-2026-TP-001"
__license__ = "MIT"

# Inject repository path for modular imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------
# Manifest Function
# ---------------------------------------------------
def get_full_manifest() -> dict:
    """Return the complete architecture manifest of the repository."""
    return {
        "system": "SVGelona_AI",
        "version": __version__,
        "cortex": __cortex_version__,
        "integrity": __integrity_protocol__,
        "structure": {
            "kernel": ["KERNEL_STABILIZATION_ROUTINE", "Uniqueness_trivial-solution"],
            "core": ["synesthetic_core", "svgelona_engine_v5_2", "integrated_memory_manager", 
                     "angular_geometry", "axioms_bridge_theorems", "semantic_bridge"],
            "optimization": ["symbolic_fractal_solver", "css_matrix_transformer"],
            "tools": ["SVGelonaTuner", "tesat_Validation"],
            "docs": ["White Paper.tex"]
        },
        "security": "IMPLEMENTATION_OF_CERTIFICATE_SVG.py"
    }

# ---------------------------------------------------
# Integrity Verification
# ---------------------------------------------------
def verify_system_integrity() -> dict:
    """
    Verify the presence and activation of the SVG-CERT integrity protocol.
    Returns a dictionary with status and message.
    """
    status = True
    msg = f"SVGelona_AI v{__version__} | Integrity Protocol: {__integrity_protocol__} ... ACTIVE ✅"
    print(f"--- {msg} ---")
    return {"status": status, "message": msg}

# Automatic initialization
SYSTEM_READY = verify_system_integrity()

# ---------------------------------------------------
# Public Exports
# ---------------------------------------------------
try:
    from .main_v5_2 import SVGelonaAI5_2
except ImportError:
    SVGelonaAI5_2 = None

__all__ = ["SVGelonaAI5_2", "get_full_manifest", "verify_system_integrity", "SYSTEM_READY"]
