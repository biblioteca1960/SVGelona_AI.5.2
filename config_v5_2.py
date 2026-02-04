"""
SVGelona_AI 5.2 - Configuracions Predefinides
Configuracions optimitzades per a diferents casos d'ús.
"""

from typing import Dict, Any, List, Optional

class SVGelonaConfig:
    """Classe base per a configuracions de SVGelona_AI."""
    
    def __init__(self, name: str, description: str, config: Dict[str, Any]):
        self.name = name
        self.description = description
        self.config = config
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Converteix la configuració a diccionari."""
        return {
            "name": self.name,
            "description": self.description,
            "config": self.config.copy()
        }
    
    def merge_with(self, custom_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fusiona amb una configuració personalitzada.
        
        Args:
            custom_config: Configuració personalitzada
            
        Returns:
            Configuració fusionada
        """
        merged = self.config.copy()
        merged.update(custom_config)
        return merged

# ============================================================================
# CONFIGURACIONS PREDEFINIDES
# ============================================================================

# ----------------------------------------------------------------------------
# CONFIGURACIÓ PER DEFECTE
# ----------------------------------------------------------------------------
DEFAULT_CONFIG = SVGelonaConfig(
    name="default",
    description="Configuració equilibrada per a ús general",
    config={
        # Paràmetres bàsics del sistema
        "max_scars": 10000,           # Límit màxim de cicatrius
        "max_fractal_depth": 12,      # Profunditat fractal màxima
        "memory_limit_mb": 100,       # Límit de memòria en MB
        
        # Mode de rendiment
        "performance_mode": "balanced",  # "performance", "balanced", "quality"
        "auto_optimize": True,           # Optimització automàtica
        "save_state_interval": 100,      # Guardar estat cada N generacions
        
        # Renderització
        "render_enabled": True,          # Habilitar renderització
        "render_quality": "medium",      # "low", "medium", "high", "ultra"
        
        # Pont semàntic
        "semantic_bridge_enabled": True,      # Habilitar pont semàntic
        "default_narrative_style": "poetic",  # "poetic", "technical", "evolutionary", "minimalist", "dramatic"
        "max_conversation_history": 50,       # Historial màxim de converses
        
        # Optimitzacions avançades
        "enable_svd_correction": True,    # Correcció SVD per a estabilitat
        "adaptive_learning": True,        # Aprenentatge adaptatiu
        "cache_enabled": True,            # Habilitar cache
        "parallel_processing": False,     # Processament paral·lel (experimental)
        
        # Paràmetres del motor fractal
        "fractal_growth_rate": 1.15,      # Taxa de creixement fractal
        "branch_pruning_threshold": 0.01, # Llindar de poda de branques
        "complexity_limit": 5000,         # Límit de complexitat
        
        # Geometria angular
        "torsion_strength": 0.1,          # Força de torsió
        "angular_damping": 0.95,          # Amortiment angular
        "phase_duration_min": 50,         # Durada mínima de fase
        "phase_duration_max": 200,        # Durada màxima de fase
        
        # Gestor de memòria
        "target_memory_utilization": 0.7, # Utilització objectiu de memòria
        "eviction_batch_size": 10,        # Mida del lot d'evicció
        "min_utility_threshold": 0.2,     # Llindar mínim d'utilitat
        
        # Sistema axiomàtic
        "min_confidence_threshold": 0.6,  # Llindar mínim de confiança
        "max_axioms_per_trauma": 3,       # Màxim d'axiomes per trauma
        "consistency_check_interval": 50, # Interval de verificació de consistència
        
        # Transformacions CSS
        "use_gpu_acceleration": True,     # Acceleració GPU per a CSS
        "optimize_for_performance": True, # Optimitzar per a rendiment
        "css_precision_digits": 6,        # Dígits de precisió CSS
        
        # Logging i monitorització
        "log_level": "INFO",              # Nivell de log
        "enable_metrics": True,           # Habilitar mètriques
        "metrics_interval": 10,           # Interval de mètriques (generacions)
        
        # Exportació
        "export_formats": ["json", "png"], # Formats d'exportació
        "auto_export": False,              # Exportació automàtica
    }
)

# ----------------------------------------------------------------------------
# CONFIGURACIÓ D'ALT RENDIMENT
# ----------------------------------------------------------------------------
PERFORMANCE_CONFIG = SVGelonaConfig(
    name="performance",
    description="Màxim rendiment, sacrificant qualitat i característiques",
    config={
        **DEFAULT_CONFIG.config,
        "performance_mode": "performance",
        "semantic_bridge_enabled": False,    # Desactivar per a més velocitat
        "max_fractal_depth": 8,              # Reduir profunditat
        "memory_limit_mb": 50,               # Menys memòria
        "render_quality": "low",             # Qualitat de renderització baixa
        "fractal_growth_rate": 1.1,          # Creixement més lent
        "branch_pruning_threshold": 0.05,    # Poda més agressiva
        "cache_enabled": True,               # Cache activada per a velocitat
        "parallel_processing": True,         # Processament paral·lel
        "angular_damping": 0.98,             # Menys amortiment (més ràpid)
        "torsion_strength": 0.05,            # Torsió més feble (més estable)
        "log_level": "WARNING",              # Menys logging
        "enable_metrics": False,             # Sense mètriques per a velocitat
    }
)

# ----------------------------------------------------------------------------
# CONFIGURACIÓ D'ALTA QUALITAT
# ----------------------------------------------------------------------------
QUALITY_CONFIG = SVGelonaConfig(
    name="quality",
    description="Màxima qualitat i detall, sacrificant rendiment",
    config={
        **DEFAULT_CONFIG.config,
        "performance_mode": "quality",
        "max_fractal_depth": 15,             # Major profunditat
        "memory_limit_mb": 200,              # Més memòria
        "render_quality": "ultra",           # Qualitat màxima
        "fractal_growth_rate": 1.2,          # Creixement més ràpid
        "branch_pruning_threshold": 0.0,     # Sense poda
        "complexity_limit": 10000,           # Límit de complexitat més alt
        "torsion_strength": 0.15,            # Torsió més forta
        "angular_damping": 0.92,             # Menys amortiment
        "css_precision_digits": 10,          # Més precisió CSS
        "save_state_interval": 50,           # Guardar més freqüentment
        "consistency_check_interval": 25,    # Verificacions més freqüents
        "auto_optimize": False,              # Optimització manual
        "parallel_processing": False,        # No paral·lel (més estable)
    }
)

# ----------------------------------------------------------------------------
# CONFIGURACIÓ CONVERSACIONAL
# ----------------------------------------------------------------------------
CONVERSATIONAL_CONFIG = SVGelonaConfig(
    name="conversational",
    description="Optimitzat per a interacció en llenguatge natural",
    config={
        **DEFAULT_CONFIG.config,
        "semantic_bridge_enabled": True,
        "default_narrative_style": "poetic",
        "max_conversation_history": 100,     # Historial més llarg
        "max_fractal_depth": 10,             # Profunditat moderada
        "render_enabled": False,             # Sense render per a velocitat
        "performance_mode": "balanced",
        "memory_limit_mb": 80,
        "adaptive_learning": True,           # Aprenentatge adaptatiu activat
        "log_level": "INFO",
        "enable_metrics": True,
        "auto_optimize": False,              # No optimitzar durant converses
        "save_state_interval": 20,           # Guardar freqüentment
        "export_formats": ["json", "txt"],   # Formats textuals
    }
)

# ----------------------------------------------------------------------------
# CONFIGURACIÓ D'INVESTIGACIÓ
# ----------------------------------------------------------------------------
RESEARCH_CONFIG = SVGelonaConfig(
    name="research",
    description="Per a experiments i investigació científica",
    config={
        **DEFAULT_CONFIG.config,
        "performance_mode": "quality",
        "max_fractal_depth": 20,             # Profunditat extrema
        "memory_limit_mb": 500,              # Molta memòria
        "complexity_limit": 20000,           # Límit alt de complexitat
        "enable_metrics": True,
        "metrics_interval": 1,               # Mètriques cada generació
        "log_level": "DEBUG",                # Logging detallat
        "auto_export": True,                 # Exportació automàtica
        "export_formats": ["json", "csv", "png", "svg"],
        "save_state_interval": 10,           # Guardar molt freqüentment
        "consistency_check_interval": 10,    # Verificacions freqüents
        "parallel_processing": False,        # No paral·lel per a consistència
        "cache_enabled": False,              # No cache per a experiments purs
        "adaptive_learning": False,          # No aprenentatge adaptatiu
        "semantic_bridge_enabled": False,    # Semàntica desactivada
    }
)

# ----------------------------------------------------------------------------
# CONFIGURACIÓ VISUAL
# ----------------------------------------------------------------------------
VISUAL_CONFIG = SVGelonaConfig(
    name="visual",
    description="Optimitzat per a visualització i renderització",
    config={
        **DEFAULT_CONFIG.config,
        "render_enabled": True,
        "render_quality": "ultra",
        "max_fractal_depth": 10,             # Profunditat equilibrada
        "css_precision_digits": 8,           # Alta precisió CSS
        "use_gpu_acceleration": True,
        "optimize_for_performance": True,
        "fractal_growth_rate": 1.25,         # Creixement ràpid
        "torsion_strength": 0.2,             # Torsió forta (efectes visuals)
        "angular_damping": 0.9,              # Poca amortiment (més dinàmic)
        "semantic_bridge_enabled": False,    # Sense semàntica
        "auto_optimize": False,              # Optimització manual
        "export_formats": ["png", "svg", "gif"],
        "auto_export": True,                 # Exportar automàticament
        "memory_limit_mb": 150,              # Més memòria per a gràfics
        "parallel_processing": False,        # No paral·lel per a estabilitat
    }
)

# ----------------------------------------------------------------------------
# CONFIGURACIÓ MINIMALISTA
# ----------------------------------------------------------------------------
MINIMAL_CONFIG = SVGelonaConfig(
    name="minimal",
    description="Configuració minimalista amb recursos mínims",
    config={
        **DEFAULT_CONFIG.config,
        "max_scars": 1000,                   # Pocs registres
        "max_fractal_depth": 5,              # Poca profunditat
        "memory_limit_mb": 25,               # Poca memòria
        "performance_mode": "performance",
        "semantic_bridge_enabled": False,
        "render_enabled": False,
        "cache_enabled": False,
        "parallel_processing": False,
        "enable_metrics": False,
        "log_level": "ERROR",                # Només errors
        "auto_optimize": False,
        "save_state_interval": 1000,         # Rarament guardar
        "branch_pruning_threshold": 0.1,     # Poda agressiva
        "complexity_limit": 1000,            # Baixa complexitat
        "angular_damping": 0.99,             # Molt amortiment (estable)
        "torsion_strength": 0.01,            # Gairebé sense torsió
    }
)

# ----------------------------------------------------------------------------
# CONFIGURACIÓ CREATIVA
# ----------------------------------------------------------------------------
CREATIVE_CONFIG = SVGelonaConfig(
    name="creative",
    description="Per a exploració creativa i generació artística",
    config={
        **DEFAULT_CONFIG.config,
        "max_fractal_depth": 12,
        "fractal_growth_rate": 1.3,          # Creixement ràpid
        "torsion_strength": 0.25,            # Torsió forta
        "angular_damping": 0.85,             # Poca amortiment
        "phase_duration_min": 20,            # Fases curtes
        "phase_duration_max": 100,           # Canvis freqüents
        "semantic_bridge_enabled": True,
        "default_narrative_style": "dramatic", # Narrativa dramàtica
        "render_quality": "high",
        "auto_optimize": True,
        "adaptive_learning": True,
        "memory_limit_mb": 120,
        "export_formats": ["png", "svg", "json"],
        "auto_export": True,
        "log_level": "INFO",
    }
)

# ============================================================================
# FUNCIONS ÚTILS
# ============================================================================

# Diccionari de totes les configuracions
ALL_CONFIGS = {
    "default": DEFAULT_CONFIG,
    "performance": PERFORMANCE_CONFIG,
    "quality": QUALITY_CONFIG,
    "conversational": CONVERSATIONAL_CONFIG,
    "research": RESEARCH_CONFIG,
    "visual": VISUAL_CONFIG,
    "minimal": MINIMAL_CONFIG,
    "creative": CREATIVE_CONFIG,
}

def get_config(config_name: str = "default") -> Dict[str, Any]:
    """
    Obté una configuració predefinida.
    
    Args:
        config_name: Nom de la configuració
        
    Returns:
        Diccionari de configuració
        
    Raises:
        ValueError: Si la configuració no existeix
    """
    if config_name not in ALL_CONFIGS:
        available = list(ALL_CONFIGS.keys())
        raise ValueError(
            f"Configuració '{config_name}' no trobada. "
            f"Disponibles: {available}"
        )
    
    return ALL_CONFIGS[config_name].config.copy()

def get_config_object(config_name: str = "default") -> SVGelonaConfig:
    """
    Obté l'objecte de configuració complet.
    
    Args:
        config_name: Nom de la configuració
        
    Returns:
        Objecte SVGelonaConfig
    """
    if config_name not in ALL_CONFIGS:
        raise ValueError(f"Configuració '{config_name}' no trobada")
    
    return ALL_CONFIGS[config_name]

def list_available_configs() -> List[str]:
    """
    Llista totes les configuracions disponibles.
    
    Returns:
        Llista de noms de configuració
    """
    return list(ALL_CONFIGS.keys())

def get_config_info(config_name: str = "default") -> Dict[str, Any]:
    """
    Obté informació sobre una configuració.
    
    Args:
        config_name: Nom de la configuració
        
    Returns:
        Diccionari amb informació
    """
    config_obj = get_config_object(config_name)
    return config_obj.to_dict()

def create_custom_config(base_config_name: str = "default", 
                        custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Crea una configuració personalitzada basant-se en una existent.
    
    Args:
        base_config_name: Configuració base
        custom_params: Paràmetres personalitzats
        
    Returns:
        Configuració personalitzada
    """
    base_config = get_config(base_config_name)
    
    if custom_params is None:
        custom_params = {}
    
    # Fusionar configuració
    merged_config = base_config.copy()
    merged_config.update(custom_params)
    
    return merged_config

def validate_config(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Valida una configuració i retorna errors/advertències.
    
    Args:
        config: Configuració a validar
        
    Returns:
        Diccionari amb errors i advertències
    """
    errors = []
    warnings = []
    
    # Paràmetres obligatoris
    required_params = ["max_scars", "max_fractal_depth", "memory_limit_mb"]
    for param in required_params:
        if param not in config:
            errors.append(f"Paràmetre obligatori '{param}' no trobat")
    
    # Valors vàlids per a performance_mode
    if "performance_mode" in config:
        valid_modes = ["performance", "balanced", "quality"]
        if config["performance_mode"] not in valid_modes:
            errors.append(f"performance_mode ha de ser un de: {valid_modes}")
    
    # Valors vàlids per a render_quality
    if "render_quality" in config:
        valid_qualities = ["low", "medium", "high", "ultra"]
        if config["render_quality"] not in valid_qualities:
            errors.append(f"render_quality ha de ser un de: {valid_qualities}")
    
    # Rangs vàlids
    if "max_fractal_depth" in config:
        depth = config["max_fractal_depth"]
        if not isinstance(depth, int) or depth < 1 or depth > 20:
            warnings.append(f"max_fractal_depth={depth} fora del rang recomanat 1-20")
    
    if "memory_limit_mb" in config:
        memory = config["memory_limit_mb"]
        if not isinstance(memory, (int, float)) or memory < 10:
            warnings.append(f"memory_limit_mb={memory} massa baix, mínim recomanat: 50")
    
    if "torsion_strength" in config:
        torsion = config["torsion_strength"]
        if torsion < 0 or torsion > 1:
            warnings.append(f"torsion_strength={torsion} fora del rang recomanat 0-1")
    
    # Compatibilitat
    if config.get("parallel_processing", False) and config.get("performance_mode") == "quality":
        warnings.append("parallel_processing pot ser inestable en mode quality")
    
    if config.get("semantic_bridge_enabled", False) and config.get("performance_mode") == "performance":
        warnings.append("semantic_bridge pot reduir el rendiment en mode performance")
    
    return {
        "errors": errors,
        "warnings": warnings,
        "is_valid": len(errors) == 0
    }

def optimize_config_for_hardware() -> Dict[str, Any]:
    """
    Optimitza la configuració automàticament basant-se en el maquinari.
    
    Returns:
        Configuració optimitzada
    """
    import psutil
    import os
    
    # Obtenir informació del sistema
    cpu_count = os.cpu_count() or 4
    memory = psutil.virtual_memory()
    total_memory_mb = memory.total / (1024 * 1024)
    
    print(f"⚙️  Detectant maquinari: {cpu_count} CPUs, {total_memory_mb:.0f} MB RAM")
    
    # Determinar configuració basant-se en recursos
    if total_memory_mb < 2000:  # Menys de 2GB
        print("  → Sistema amb recursos limitats, usant configuració minimal")
        base_config = "minimal"
        
    elif total_memory_mb < 8000:  # Menys de 8GB
        print("  → Sistema amb recursos moderats, usant configuració equilibrada")
        base_config = "default"
        
    else:  # 8GB o més
        if cpu_count >= 8:
            print("  → Sistema potent amb múltiples CPUs, usant configuració d'alt rendiment")
            base_config = "performance"
        else:
            print("  → Sistema amb molta RAM, usant configuració d'alta qualitat")
            base_config = "quality"
    
    # Obtener configuració base
    config = get_config(base_config)
    
    # Ajustar basant-se en memòria disponible
    if base_config != "minimal":
        # Utilitzar el 40% de la memòria disponible, amb límits
        target_memory = min(total_memory_mb * 0.4, 2000)  # Màxim 2GB
        config["memory_limit_mb"] = max(50, int(target_memory))
        print(f"  → Memòria assignada: {config['memory_limit_mb']} MB")
    
    # Ajustar basant-se en CPUs
    if cpu_count >= 4 and base_config != "minimal":
        config["parallel_processing"] = True
        print(f"  → Processament paral·lel activat ({cpu_count} CPUs)")
    
    return config

def save_config_to_file(config: Dict[str, Any], filename: str = "svgelona_config.json"):
    """
    Guarda una configuració a un fitxer.
    
    Args:
        config: Configuració a guardar
        filename: Nom del fitxer
    """
    import json
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Configuració guardada a '{filename}'")

def load_config_from_file(filename: str = "svgelona_config.json") -> Dict[str, Any]:
    """
    Carrega una configuració des d'un fitxer.
    
    Args:
        filename: Nom del fitxer
        
    Returns:
        Configuració carregada
    """
    import json
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"📂 Configuració carregada des de '{filename}'")
        return config
        
    except FileNotFoundError:
        print(f"⚠️  Fitxer '{filename}' no trobat, usant configuració per defecte")
        return get_config("default")
    except json.JSONDecodeError as e:
        print(f"⚠️  Error llegint '{filename}': {e}, usant configuració per defecte")
        return get_config("default")

# ============================================================================
# INTERFÍCIE DE LÍNIA DE COMANDES
# ============================================================================

def print_config_summary(config_name: str = "default"):
    """
    Imprimeix un resum d'una configuració.
    
    Args:
        config_name: Nom de la configuració
    """
    config_obj = get_config_object(config_name)
    info = config_obj.to_dict()
    
    print(f"\n{'='*60}")
    print(f"CONFIGURACIÓ: {info['name'].upper()}")
    print(f"{'='*60}")
    print(f"Descripció: {info['description']}")
    print(f"\nParàmetres principals:")
    
    # Agrupar paràmetres per categoria
    categories = {
        "Sistema Bàsic": ["max_scars", "max_fractal_depth", "memory_limit_mb"],
        "Rendiment": ["performance_mode", "auto_optimize", "parallel_processing"],
        "Renderització": ["render_enabled", "render_quality", "use_gpu_acceleration"],
        "Pont Semàntic": ["semantic_bridge_enabled", "default_narrative_style", "max_conversation_history"],
        "Geometria": ["torsion_strength", "angular_damping", "phase_duration_min", "phase_duration_max"],
        "Optimització": ["branch_pruning_threshold", "complexity_limit", "cache_enabled"],
    }
    
    config = info["config"]
    
    for category, params in categories.items():
        print(f"\n  {category}:")
        for param in params:
            if param in config:
                value = config[param]
                print(f"    • {param}: {value}")
    
    print(f"\n{'='*60}")

def main():
    """Funció principal per a interacció des de la línia de comandes."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Gestor de configuracions per a SVGelona_AI 5.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'ús:
  %(prog)s list                            # Llista totes les configuracions
  %(prog)s show performance                # Mostra la configuració d'alt rendiment
  %(prog)s create myconfig --base quality  # Crea configuració personalitzada
  %(prog)s validate myconfig.json          # Valida un fitxer de configuració
  %(prog)s optimize                        # Genera configuració optimitzada per al teu maquinari
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Comanda a executar")
    
    # Comanda: list
    list_parser = subparsers.add_parser("list", help="Llista configuracions disponibles")
    
    # Comanda: show
    show_parser = subparsers.add_parser("show", help="Mostra una configuració")
    show_parser.add_argument("config_name", help="Nom de la configuració")
    
    # Comanda: create
    create_parser = subparsers.add_parser("create", help="Crea configuració personalitzada")
    create_parser.add_argument("output_file", help="Fitxer de sortida")
    create_parser.add_argument("--base", default="default", help="Configuració base")
    create_parser.add_argument("--params", help="Paràmetres JSON personalitzats")
    
    # Comanda: validate
    validate_parser = subparsers.add_parser("validate", help="Valida una configuració")
    validate_parser.add_argument("config_file", help="Fitxer de configuració")
    
    # Comanda: optimize
    optimize_parser = subparsers.add_parser("optimize", help="Optimitza per al maquinari")
    optimize_parser.add_argument("--output", help="Fitxer de sortida (opcional)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "list":
            print("📋 Configuracions disponibles:\n")
            for name in list_available_configs():
                config_obj = get_config_object(name)
                print(f"  • {name:15} - {config_obj.description}")
        
        elif args.command == "show":
            print_config_summary(args.config_name)
        
        elif args.command == "create":
            base_config = get_config(args.base)
            
            # Parsejar paràmetres personalitzats si n'hi ha
            custom_params = {}
            if args.params:
                import json
                custom_params = json.loads(args.params)
            
            # Fusionar configuracions
            final_config = base_config.copy()
            final_config.update(custom_params)
            
            # Guardar a fitxer
            save_config_to_file(final_config, args.output_file)
            
            # Validar
            validation = validate_config(final_config)
            if validation["warnings"]:
                print("\n⚠️  Advertències:")
                for warning in validation["warnings"]:
                    print(f"  • {warning}")
        
        elif args.command == "validate":
            config = load_config_from_file(args.config_file)
            validation = validate_config(config)
            
            if validation["is_valid"]:
                print("✅ Configuració vàlida!")
            else:
                print("❌ Configuració invàlida:")
                for error in validation["errors"]:
                    print(f"  • {error}")
            
            if validation["warnings"]:
                print("\n⚠️  Advertències:")
                for warning in validation["warnings"]:
                    print(f"  • {warning}")
        
        elif args.command == "optimize":
            config = optimize_config_for_hardware()
            
            if args.output:
                save_config_to_file(config, args.output)
                print(f"\n✨ Configuració optimitzada guardada a '{args.output}'")
            else:
                print("\n✨ Configuració optimitzada per al teu maquinari:")
                for key, value in list(config.items())[:10]:  # Mostrar primers 10
                    print(f"  {key}: {value}")
                print(f"  ... i {len(config)-10} paràmetres més")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()