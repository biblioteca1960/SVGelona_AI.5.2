"""
SVGelona_AI 5.2 - Mòduls Principals
Sistema d'IA generativa fractal amb gestió de memòria i pont semàntic.
"""

# Exposar les classes principals per a importació fàcil
from .svgelona_engine_v5_2 import (
    SVGelonaEngine,
    FractalState,
    EvolutionaryScar,
    OptimizedScarArchive,
    OptimizedFractalModule
)

from .integrated_memory_manager import (
    IntegratedMemoryManager,
    MemoryPriority,
    MemoryItem
)

from .axioms_bridge_theorems import (
    AxiomBridgeEngine,
    Axiom,
    AxiomCategory,
    Theorem
)

from .angular_geometry import (
    AngularGeometryEngine,
    AngularState,
    AngularPhase
)

from .semantic_bridge import (
    SemanticBridge,
    NarrativeStyle,
    IntentCategory,
    SemanticMapping,
    IntentInterpretation
)

# Versió del mòdul
__version__ = "5.2.0"
__author__ = "SVGelona_AI Team"
__description__ = "Sistema d'IA generativa fractal amb gestió de memòria integrada i pont semàntic"

# Llistat públic d'exportacions
__all__ = [
    # Motor principal
    "SVGelonaEngine",
    "FractalState",
    "EvolutionaryScar",
    "OptimizedScarArchive",
    "OptimizedFractalModule",
    
    # Gestor de memòria
    "IntegratedMemoryManager",
    "MemoryPriority",
    "MemoryItem",
    
    # Sistema axiomàtic
    "AxiomBridgeEngine",
    "Axiom",
    "AxiomCategory",
    "Theorem",
    
    # Geometria angular
    "AngularGeometryEngine",
    "AngularState",
    "AngularPhase",
    
    # Pont semàntic
    "SemanticBridge",
    "NarrativeStyle",
    "IntentCategory",
    "SemanticMapping",
    "IntentInterpretation",
]

# Funció per obtenir informació del mòdul
def get_module_info() -> dict:
    """
    Retorna informació sobre el mòdul core.
    
    Returns:
        Diccionari amb informació del mòdul
    """
    return {
        "module": "core",
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "components": {
            "engine": "SVGelonaEngine - Motor principal d'evolució fractal",
            "memory_manager": "IntegratedMemoryManager - Gestor de memòria amb poda selectiva",
            "axiom_bridge": "AxiomBridgeEngine - Sistema axiomàtic pont-teoremes",
            "angular_geometry": "AngularGeometryEngine - Geometria angular amb estabilització SVD",
            "semantic_bridge": "SemanticBridge - Pont semàntic entre geometria i llenguatge natural"
        },
        "exports": __all__
    }

# Inicialització del mòdul
def init_core_module():
    """Inicialitza els components del mòdul core."""
    print(f"Inicialitzant SVGelona_AI Core v{__version__}")
    print(f"Components disponibles: {len(__all__)}")
    
    return {
        "status": "ready",
        "version": __version__,
        "components_loaded": len(__all__)
    }

# Inicialització automàtica (opcional)
if __name__ != "__main__":
    # Registre de mòdul carregat (per a debugging)
    import sys
    if "core" not in sys.modules:
        sys.modules["core"] = sys.modules[__name__]