"""
SVGelona_AI 5.2 - Mòduls d'Optimització
Tècniques avançades d'optimització per a generació fractal.
"""

from .symbolic_fractal_solver import (
    SymbolicFractalSolver,
    FractalEquation,
    OptimizationResult
)

from .css_matrix_transformer import (
    CSSMatrixTransformer,
    CSSTransform
)

# Versió del mòdul
__version__ = "5.2.0"
__author__ = "SVGelona_AI Team"
__description__ = "Mòduls d'optimització per a generació fractal i transformacions CSS"

# Llistat públic d'exportacions
__all__ = [
    # Solucionador simbòlic
    "SymbolicFractalSolver",
    "FractalEquation",
    "OptimizationResult",
    
    # Transformador CSS
    "CSSMatrixTransformer",
    "CSSTransform",
]

# Funció per obtenir informació del mòdul
def get_module_info() -> dict:
    """
    Retorna informació sobre el mòdul d'optimització.
    
    Returns:
        Diccionari amb informació del mòdul
    """
    return {
        "module": "optimization",
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "components": {
            "symbolic_solver": "SymbolicFractalSolver - Optimització de fractals mitjançant anàlisi simbòlica",
            "css_transformer": "CSSMatrixTransformer - Transformacions CSS optimitzades per GPU"
        },
        "exports": __all__
    }

# Inicialització del mòdul
def init_optimization_module():
    """Inicialitza els components del mòdul d'optimització."""
    print(f"Inicialitzant SVGelona_AI Optimization v{__version__}")
    print(f"Components disponibles: {len(__all__)}")
    
    return {
        "status": "ready",
        "version": __version__,
        "components_loaded": len(__all__)
    }