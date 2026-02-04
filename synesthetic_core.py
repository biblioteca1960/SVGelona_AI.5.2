"""
SVGelona_AI 5.3 - Nucli Sinestèsic
Arquitectura de pensament geomètrico-semàntic integrat.
El sistema pensa i actua en ambdós dominis simultàniament.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import threading
import queue
import asyncio
import json
import math
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

# ============================================================================
# ESTRUCTURES FONAMENTALS
# ============================================================================

class ThoughtMode(Enum):
    """Modalitats de pensament del sistema."""
    GEOMETRIC = auto()      # Pensament pur en tensors i fractals
    SEMANTIC = auto()       # Pensament pur en conceptes i narracions
    SYNESTHETIC = auto()    # Pensament híbrid simultani
    INTUITIVE = auto()      # Pensament intuïtiu emergent
    CRITICAL = auto()       # Pensament crític i auto-reflexiu

class ConceptualPrimitive(Enum):
    """Primitives conceptuals fonamentals."""
    ORDER = "ordre"
    CHAOS = "caos"
    SYMMETRY = "simetria"
    ASYMMETRY = "asimetria"
    STABILITY = "estabilitat"
    INSTABILITY = "inestabilitat"
    COMPLEXITY = "complexitat"
    SIMPLICITY = "simplicitat"
    TENSION = "tensió"
    RELEASE = "alliberament"
    EMERGENCE = "emergència"
    DISSOLUTION = "dissolució"
    ROTATION = "rotació"
    TRANSLATION = "translació"
    EXPANSION = "expansió"
    CONTRACTION = "contracció"

@dataclass
class SynestheticThought:
    """Unit bàsica de pensament sinestèsic."""
    
    thought_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Components geomètrics
    geometric_vector: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torsion_tensor: Optional[np.ndarray] = None
    fractal_seed: Optional[np.ndarray] = None
    
    # Components semàntics
    semantic_concepts: List[ConceptualPrimitive] = field(default_factory=list)
    narrative_fragment: str = ""
    emotional_valence: float = 0.0  # -1 (negatiu) a +1 (positiu)
    
    # Metacognició
    confidence: float = 0.5
    coherence_score: float = 0.0  # Coherència entre components
    mode: ThoughtMode = ThoughtMode.SYNESTHETIC
    
    def __post_init__(self):
        if self.torsion_tensor is None:
            self.torsion_tensor = np.eye(3)
        if self.fractal_seed is None:
            self.fractal_seed = np.random.normal(0, 1, 3)
    
    def calculate_coherence(self) -> float:
        """Calcula la coherència entre components geomètrics i semàntics."""
        if not self.semantic_concepts:
            return 0.0
        
        # Mapeig concepte → patró geomètric esperat
        concept_patterns = {
            ConceptualPrimitive.ORDER: np.array([0.9, 0.1, 0.0]),
            ConceptualPrimitive.CHAOS: np.array([0.1, 0.9, 0.0]),
            ConceptualPrimitive.SYMMETRY: np.array([0.5, 0.5, 0.0]),
            ConceptualPrimitive.TENSION: np.array([0.7, 0.3, 0.5]),
            ConceptualPrimitive.RELEASE: np.array([0.3, 0.7, 0.2]),
        }
        
        # Calcular vector semàntic mitjà
        semantic_vector = np.zeros(3)
        for concept in self.semantic_concepts:
            if concept in concept_patterns:
                semantic_vector += concept_patterns[concept]
        
        if len(self.semantic_concepts) > 0:
            semantic_vector /= len(self.semantic_concepts)
        
        # Comparar amb vector geomètric
        if np.linalg.norm(semantic_vector) > 0 and np.linalg.norm(self.geometric_vector) > 0:
            cos_similarity = np.dot(semantic_vector, self.geometric_vector) / (
                np.linalg.norm(semantic_vector) * np.linalg.norm(self.geometric_vector)
            )
            return float((cos_similarity + 1) / 2)  # Normalitzar a [0, 1]
        
        return 0.0
    
    def evolve(self, engine_ref) -> 'SynestheticThought':
        """Evoluciona el pensament interactuant amb el motor."""
        # Aplicar torsió al vector geomètric
        if self.torsion_tensor is not None:
            self.geometric_vector = self.geometric_vector @ self.torsion_tensor
        
        # Evolucionar fractal
        if engine_ref and hasattr(engine_ref, 'fractal_module'):
            fractal = engine_ref.fractal_module.spawn_fractal_manifold(
                self.fractal_seed,
                depth=3 + int(self.confidence * 3)
            )
            
            # Actualitzar concepts basant-se en fractal
            if fractal["entropy"] > 0.7:
                self.semantic_concepts.append(ConceptualPrimitive.CHAOS)
            elif fractal["entropy"] < 0.3:
                self.semantic_concepts.append(ConceptualPrimitive.ORDER)
        
        # Recalcular coherència
        self.coherence_score = self.calculate_coherence()
        
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Converteix a diccionari per a serialització."""
        return {
            "id": self.thought_id,
            "timestamp": self.timestamp.isoformat(),
            "geometric": {
                "vector": self.geometric_vector.tolist(),
                "torsion": self.torsion_tensor.tolist() if self.torsion_tensor is not None else None,
                "fractal_seed": self.fractal_seed.tolist() if self.fractal_seed is not None else None
            },
            "semantic": {
                "concepts": [c.value for c in self.semantic_concepts],
                "narrative": self.narrative_fragment,
                "valence": self.emotional_valence
            },
            "metacognition": {
                "confidence": self.confidence,
                "coherence": self.coherence_score,
                "mode": self.mode.name
            }
        }

# ============================================================================
# MEMÒRIA SEMÀNTICO-GEOMÈTRICA
# ============================================================================

@dataclass
class SynestheticMemoryCell:
    """Cèl·lula de memòria que emmagatzema patrons sinestèsics."""
    
    pattern_id: str
    geometric_pattern: np.ndarray  # Patró geomètric (3D o tensor)
    semantic_pattern: List[ConceptualPrimitive]  # Patró semàntic
    narrative_template: str  # Plantilla narrativa
    activation_count: int = 0
    last_activated: datetime = field(default_factory=datetime.now)
    associative_strength: float = 1.0  # Força associativa
    
    def matches(self, 
                geometric_input: np.ndarray, 
                semantic_input: List[ConceptualPrimitive],
                threshold: float = 0.7) -> bool:
        """Determina si un input coincideix amb aquest patró."""
        
        # Similitud geomètrica
        geometric_sim = self._calculate_geometric_similarity(geometric_input)
        
        # Similitud semàntica (sobreposició de Jaccard)
        semantic_sim = self._calculate_semantic_similarity(semantic_input)
        
        # Combinar (ponderació adaptativa)
        combined_sim = (geometric_sim * 0.6 + semantic_sim * 0.4)
        
        return combined_sim >= threshold
    
    def _calculate_geometric_similarity(self, other_vector: np.ndarray) -> float:
        """Calcula similitud entre vectors geomètrics."""
        if np.linalg.norm(self.geometric_pattern) == 0 or np.linalg.norm(other_vector) == 0:
            return 0.0
        
        cos_sim = np.dot(self.geometric_pattern, other_vector) / (
            np.linalg.norm(self.geometric_pattern) * np.linalg.norm(other_vector)
        )
        
        return float((cos_sim + 1) / 2)  # Normalitzar a [0, 1]
    
    def _calculate_semantic_similarity(self, other_concepts: List[ConceptualPrimitive]) -> float:
        """Calcula similitud entre conjunts de concepts."""
        if not self.semantic_pattern or not other_concepts:
            return 0.0
        
        set_self = set(self.semantic_pattern)
        set_other = set(other_concepts)
        
        intersection = len(set_self.intersection(set_other))
        union = len(set_self.union(set_other))
        
        return intersection / union if union > 0 else 0.0
    
    def activate(self):
        """Activa la cèl·lula de memòria."""
        self.activation_count += 1
        self.last_activated = datetime.now()
        self.associative_strength = min(2.0, self.associative_strength * 1.1)  # Reforçar
    
    def decay(self, decay_rate: float = 0.99):
        """Aplica decaïment a la força associativa."""
        self.associative_strength *= decay_rate

# ============================================================================
# CORTEX SINESTÈSIC
# ============================================================================

class SynestheticCortex:
    """
    Còrtex que processa pensaments en temps real.
    Implementa fluxos de pensament paral·lels i interacció asíncrona.
    """
    
    def __init__(self, engine_ref, config: Optional[Dict[str, Any]] = None):
        self.engine = engine_ref
        self.thought_stream = queue.Queue()  # Flux de pensaments
        self.memory_cells: Dict[str, SynestheticMemoryCell] = {}
        
        # Fluxos de pensament paral·lels
        self.thought_streams = {
            ThoughtMode.GEOMETRIC: queue.Queue(),
            ThoughtMode.SEMANTIC: queue.Queue(),
            ThoughtMode.SYNESTHETIC: queue.Queue(),
            ThoughtMode.INTUITIVE: queue.Queue(),
            ThoughtMode.CRITICAL: queue.Queue(),
        }
        
        # Configuració
        self.config = {
            "thought_generation_interval": 0.1,  # Segons entre pensaments
            "max_concurrent_thoughts": 5,
            "memory_capacity": 1000,
            "activation_threshold": 0.7,
            "enable_intuitive_leaps": True,
            "enable_critique": True,
            "stream_processing": True,
        }
        
        if config:
            self.config.update(config)
        
        # Executors per a processament paral·lel
        self.executor = ThreadPoolExecutor(max_workers=self.config["max_concurrent_thoughts"])
        self.processing = False
        self.current_thoughts: List[SynestheticThought] = []
        
        # Inicialitzar memòria amb patrons bàsics
        self._initialize_memory()
        
        # Estadístiques
        self.stats = {
            "thoughts_generated": 0,
            "memory_activations": 0,
            "intuitive_leaps": 0,
            "coherence_violations": 0,
            "stream_overflows": 0,
        }
    
    def _initialize_memory(self):
        """Inicialitza la memòria amb patrons bàsics."""
        
        # Patrons bàsics orden-caos
        basic_patterns = [
            {
                "geometric": np.array([0.9, 0.1, 0.0]),
                "semantic": [ConceptualPrimitive.ORDER, ConceptualPrimitive.STABILITY],
                "narrative": "Equilibri perfecte, ordre immutable"
            },
            {
                "geometric": np.array([0.1, 0.9, 0.0]),
                "semantic": [ConceptualPrimitive.CHAOS, ConceptualPrimitive.INSTABILITY],
                "narrative": "Caos creatiu, desordre generatiu"
            },
            {
                "geometric": np.array([0.5, 0.5, 0.5]),
                "semantic": [ConceptualPrimitive.TENSION, ConceptualPrimitive.COMPLEXITY],
                "narrative": "Tensió productiva entre forces oposades"
            },
            {
                "geometric": np.array([0.7, 0.3, 0.8]),
                "semantic": [ConceptualPrimitive.EMERGENCE, ConceptualPrimitive.EXPANSION],
                "narrative": "Emergència de noves estructures"
            },
            {
                "geometric": np.array([0.3, 0.7, 0.2]),
                "semantic": [ConceptualPrimitive.DISSOLUTION, ConceptualPrimitive.CONTRACTION],
                "narrative": "Dissolució en components bàsics"
            },
        ]
        
        for i, pattern in enumerate(basic_patterns):
            cell_id = f"pattern_{i}"
            self.memory_cells[cell_id] = SynestheticMemoryCell(
                pattern_id=cell_id,
                geometric_pattern=pattern["geometric"],
                semantic_pattern=pattern["semantic"],
                narrative_template=pattern["narrative"]
            )
    
    def start_thinking(self):
        """Inicia el procés de pensament continu."""
        self.processing = True
        
        # Iniciar fluxos paral·lels
        if self.config["stream_processing"]:
            for mode in ThoughtMode:
                threading.Thread(
                    target=self._process_thought_stream,
                    args=(mode,),
                    daemon=True
                ).start()
        
        # Pensament principal
        threading.Thread(target=self._generate_continuous_thoughts, daemon=True).start()
        
        # Auto-crítica periòdica
        if self.config["enable_critique"]:
            threading.Thread(target=self._critical_analysis_loop, daemon=True).start()
        
        print(f"🧠 Cortex sinestèsic iniciat: {len(self.memory_cells)} patrons inicials")
    
    def stop_thinking(self):
        """Atura el procés de pensament."""
        self.processing = False
    
    def _generate_continuous_thoughts(self):
        """Genera pensaments contínuament."""
        thought_counter = 0
        
        while self.processing:
            try:
                # Generar nou pensament
                thought = self._generate_thought(thought_counter)
                thought_counter += 1
                
                # Posar en flux principal
                self.thought_stream.put(thought)
                
                # Distribuir als fluxos específics
                self.thought_streams[thought.mode].put(thought)
                
                # Guardar
                self.current_thoughts.append(thought)
                if len(self.current_thoughts) > 50:
                    self.current_thoughts = self.current_thoughts[-50:]
                
                self.stats["thoughts_generated"] += 1
                
                # Interval
                time.sleep(self.config["thought_generation_interval"])
                
            except Exception as e:
                print(f"Error generant pensament: {e}")
                time.sleep(1)
    
    def _generate_thought(self, index: int) -> SynestheticThought:
        """Genera un nou pensament."""
        
        # Decidir mode de pensament
        mode_weights = {
            ThoughtMode.GEOMETRIC: 0.2,
            ThoughtMode.SEMANTIC: 0.2,
            ThoughtMode.SYNESTHETIC: 0.4,
            ThoughtMode.INTUITIVE: 0.1,
            ThoughtMode.CRITICAL: 0.1,
        }
        
        mode = np.random.choice(
            list(mode_weights.keys()),
            p=list(mode_weights.values())
        )
        
        # Generar components basant-se en l'estat actual del sistema
        if self.engine and hasattr(self.engine, 'angular_geometry'):
            angular_state = self.engine.angular_geometry.state
            geometric_vector = angular_state.angular_momentum.copy()
            torsion_tensor = angular_state.torsion_tensor.copy()
        else:
            geometric_vector = np.random.normal(0, 1, 3)
            torsion_tensor = np.eye(3) + np.random.normal(0, 0.1, (3, 3))
        
        # Consultar memòria per a concepts
        semantic_concepts = self._retrieve_semantic_concepts(geometric_vector)
        
        # Generar fragment narratiu
        narrative = self._generate_narrative_fragment(semantic_concepts, geometric_vector)
        
        # Crear pensament
        thought = SynestheticThought(
            thought_id=f"thought_{index}_{datetime.now().timestamp()}",
            geometric_vector=geometric_vector,
            torsion_tensor=torsion_tensor,
            fractal_seed=geometric_vector.copy(),
            semantic_concepts=semantic_concepts,
            narrative_fragment=narrative,
            emotional_valence=np.random.uniform(-0.5, 0.5),
            confidence=0.5 + np.random.uniform(-0.2, 0.2),
            mode=mode
        )
        
        # Evolucionar i calcular coherència
        thought = thought.evolve(self.engine)
        
        return thought
    
    def _retrieve_semantic_concepts(self, geometric_vector: np.ndarray) -> List[ConceptualPrimitive]:
        """Recupera concepts semàntics de la memòria basant-se en un patró geomètric."""
        
        if not self.memory_cells:
            # Retornar concepts bàsics si no hi ha memòria
            if np.linalg.norm(geometric_vector) < 0.5:
                return [ConceptualPrimitive.ORDER]
            else:
                return [ConceptualPrimitive.CHAOS]
        
        # Buscar cèl·lules de memòria que coincideixin
        matched_concepts = []
        empty_concepts = []  # Concepts sense input semàntic
        
        for cell_id, cell in self.memory_cells.items():
            if cell.matches(geometric_vector, []):
                # Activar cèl·lula
                cell.activate()
                self.stats["memory_activations"] += 1
                
                # Afegir concepts (evitar duplicats)
                for concept in cell.semantic_pattern:
                    if concept not in matched_concepts:
                        matched_concepts.append(concept)
        
        # Si no hi ha coincidències, crear intuïtivament
        if not matched_concepts and self.config["enable_intuitive_leaps"]:
            intuitive_concepts = self._intuitive_concept_generation(geometric_vector)
            matched_concepts.extend(intuitive_concepts)
            self.stats["intuitive_leaps"] += 1
        
        return matched_concepts
    
    def _intuitive_concept_generation(self, geometric_vector: np.ndarray) -> List[ConceptualPrimitive]:
        """Genera concepts de manera intuïtiva."""
        
        concepts = []
        
        # Analitzar característiques del vector
        magnitude = np.linalg.norm(geometric_vector)
        variability = np.std(geometric_vector)
        
        if magnitude < 0.3:
            concepts.append(ConceptualPrimitive.STABILITY)
            concepts.append(ConceptualPrimitive.SIMPLICITY)
        elif magnitude > 0.7:
            concepts.append(ConceptualPrimitive.INSTABILITY)
            concepts.append(ConceptualPrimitive.COMPLEXITY)
        
        if variability > 0.5:
            concepts.append(ConceptualPrimitive.CHAOS)
            concepts.append(ConceptualPrimitive.ASYMMETRY)
        elif variability < 0.2:
            concepts.append(ConceptualPrimitive.ORDER)
            concepts.append(ConceptualPrimitive.SYMMETRY)
        
        # Afegir aleatorietat creativa
        if np.random.random() < 0.3:
            all_concepts = list(ConceptualPrimitive)
            random_concept = np.random.choice(all_concepts)
            if random_concept not in concepts:
                concepts.append(random_concept)
        
        return concepts
    
    def _generate_narrative_fragment(self, 
                                   concepts: List[ConceptualPrimitive],
                                   geometric_vector: np.ndarray) -> str:
        """Genera un fragment narratiu basat en concepts i geometria."""
        
        if not concepts:
            return "Silenci conceptual..."
        
        # Plantilles narratives basades en combinacions de concepts
        narrative_templates = {
            frozenset([ConceptualPrimitive.ORDER, ConceptualPrimitive.STABILITY]): 
                "Harmonia perfecta, equilibri immutable.",
            
            frozenset([ConceptualPrimitive.CHAOS, ConceptualPrimitive.INSTABILITY]): 
                "Turbulència creativa, ordre en descomposició.",
            
            frozenset([ConceptualPrimitive.TENSION, ConceptualPrimitive.COMPLEXITY]): 
                "Forces oposades en equilibri precari.",
            
            frozenset([ConceptualPrimitive.EMERGENCE, ConceptualPrimitive.EXPANSION]): 
                "Novetat que sorgeix de la interacció.",
            
            frozenset([ConceptualPrimitive.DISSOLUTION, ConceptualPrimitive.CONTRACTION]): 
                "Retorn a l'essencial, alliberament de forma.",
            
            frozenset([ConceptualPrimitive.SYMMETRY, ConceptualPrimitive.ORDER]): 
                "Bellesa geomètrica, proporció perfecta.",
            
            frozenset([ConceptualPrimitive.ASYMMETRY, ConceptualPrimitive.CHAOS]): 
                "Ruptura de patrons, bellesa inesperada.",
        }
        
        concept_set = frozenset(concepts)
        
        if concept_set in narrative_templates:
            return narrative_templates[concept_set]
        
        # Si no hi ha plantilla exacta, generar combinant
        if len(concepts) >= 2:
            primary = concepts[0].value
            secondary = concepts[1].value
            return f"{primary.capitalize()} en diàleg amb {secondary}."
        
        # Últim recurs
        return f"Experiència de {concepts[0].value}."
    
    def _process_thought_stream(self, mode: ThoughtMode):
        """Processa un flux específic de pensaments."""
        while self.processing:
            try:
                thought = self.thought_streams[mode].get(timeout=1)
                
                # Processament específic del mode
                if mode == ThoughtMode.GEOMETRIC:
                    self._process_geometric_thought(thought)
                elif mode == ThoughtMode.SEMANTIC:
                    self._process_semantic_thought(thought)
                elif mode == ThoughtMode.SYNESTHETIC:
                    self._process_synesthetic_thought(thought)
                elif mode == ThoughtMode.INTUITIVE:
                    self._process_intuitive_thought(thought)
                elif mode == ThoughtMode.CRITICAL:
                    self._process_critical_thought(thought)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processant flux {mode}: {e}")
    
    def _process_synesthetic_thought(self, thought: SynestheticThought):
        """Processa un pensament sinestèsic (integració completa)."""
        
        # Verificar coherència
        if thought.coherence_score < 0.3:
            self.stats["coherence_violations"] += 1
            
            # Intentar reconciliar
            thought = self._reconcile_incoherence(thought)
        
        # Aprendre nou patró si és prou únic
        if thought.coherence_score > 0.7:
            self._learn_pattern(thought)
        
        # Si la coherència és alta, aplicar al motor
        if thought.coherence_score > 0.8 and self.engine:
            self._apply_to_engine(thought)
    
    def _reconcile_incoherence(self, thought: SynestheticThought) -> SynestheticThought:
        """Intentar reconciliar un pensament incoherent."""
        
        # Estratègia 1: Ajustar components semàntics
        if np.linalg.norm(thought.geometric_vector) > 0:
            new_concepts = self._retrieve_semantic_concepts(thought.geometric_vector)
            thought.semantic_concepts = new_concepts
        
        # Estratègia 2: Generar nova narrativa
        thought.narrative_fragment = self._generate_narrative_fragment(
            thought.semantic_concepts,
            thought.geometric_vector
        )
        
        # Recalcular
        thought.coherence_score = thought.calculate_coherence()
        
        return thought
    
    def _learn_pattern(self, thought: SynestheticThought):
        """Aprèn un nou patró de la memòria."""
        
        if len(self.memory_cells) >= self.config["memory_capacity"]:
            # Aplicar decaïment i eliminar més febles
            self._prune_memory()
        
        # Crear nova cèl·lula
        pattern_id = f"learned_{len(self.memory_cells)}_{datetime.now().timestamp()}"
        
        # Normalitzar patró geomètric
        if np.linalg.norm(thought.geometric_vector) > 0:
            geometric_pattern = thought.geometric_vector / np.linalg.norm(thought.geometric_vector)
        else:
            geometric_pattern = thought.geometric_vector.copy()
        
        new_cell = SynestheticMemoryCell(
            pattern_id=pattern_id,
            geometric_pattern=geometric_pattern,
            semantic_pattern=thought.semantic_concepts.copy(),
            narrative_template=thought.narrative_fragment,
            associative_strength=1.0
        )
        
        self.memory_cells[pattern_id] = new_cell
    
    def _prune_memory(self, target_size: int = 800):
        """Podar memòria, mantenint les cèl·lules més actives."""
        
        if len(self.memory_cells) <= target_size:
            return
        
        # Ordenar per activació i força
        cells_to_keep = sorted(
            self.memory_cells.values(),
            key=lambda x: (x.activation_count, x.associative_strength),
            reverse=True
        )[:target_size]
        
        # Reconstruir diccionari
        self.memory_cells = {cell.pattern_id: cell for cell in cells_to_keep}
    
    def _apply_to_engine(self, thought: SynestheticThought):
        """Aplica un pensament coherent al motor."""
        
        if not self.engine:
            return
        
        try:
            # Aplicar a geometria angular
            if hasattr(self.engine, 'angular_geometry'):
                self.engine.angular_geometry.state.angular_momentum = (
                    thought.geometric_vector * 0.3 + 
                    self.engine.angular_geometry.state.angular_momentum * 0.7
                )
                
                # Aplicar torsió suau
                if thought.torsion_tensor is not None:
                    self.engine.angular_geometry.state.torsion_tensor = (
                        thought.torsion_tensor * 0.1 +
                        self.engine.angular_geometry.state.torsion_tensor * 0.9
                    )
            
            # Aplicar al motor fractal
            if hasattr(self.engine, 'fractal_module'):
                self.engine.fractal_module.max_depth = min(
                    15, int(8 + thought.confidence * 7)
                )
            
            # Registrar en l'historial de pensaments aplicats
            if not hasattr(self.engine, 'applied_thoughts'):
                self.engine.applied_thoughts = []
            
            self.engine.applied_thoughts.append({
                "timestamp": datetime.now().isoformat(),
                "thought": thought.to_dict(),
                "coherence": thought.coherence_score
            })
            
        except Exception as e:
            print(f"Error aplicant pensament al motor: {e}")
    
    def _critical_analysis_loop(self):
        """Bucle d'anàlisi crítica periòdica."""
        
        while self.processing:
            time.sleep(10)  # Cada 10 segons
            
            if not self.current_thoughts:
                continue
            
            # Analitzar tendències
            coherence_scores = [t.coherence_score for t in self.current_thoughts[-20:]]
            avg_coherence = np.mean(coherence_scores) if coherence_scores else 0
            
            # Generar pensament crític
            critical_thought = SynestheticThought(
                thought_id=f"critical_{datetime.now().timestamp()}",
                semantic_concepts=[ConceptualPrimitive.TENSION, ConceptualPrimitive.COMPLEXITY],
                narrative_fragment=f"Auto-anàlisi: coherència mitjana {avg_coherence:.2f}",
                confidence=0.8,
                mode=ThoughtMode.CRITICAL
            )
            
            # Posar en flux crític
            self.thought_streams[ThoughtMode.CRITICAL].put(critical_thought)
    
    def interact(self, user_input: str) -> Dict[str, Any]:
        """
        Interacció en temps real amb l'usuari.
        
        Args:
            user_input: Entrada de l'usuari
            
        Returns:
            Resposta del sistema
        """
        
        # Processar input
        input_concepts = self._parse_user_input(user_input)
        
        # Generar resposta pensada
        response_thought = self._generate_response_thought(input_concepts, user_input)
        
        # Evolucionar resposta
        response_thought = response_thought.evolve(self.engine)
        
        # Generar resposta narrativa
        narrative = self._generate_interaction_narrative(response_thought, user_input)
        
        # Aplicar si és coherent
        if response_thought.coherence_score > 0.6:
            self._apply_to_engine(response_thought)
        
        return {
            "user_input": user_input,
            "system_response": narrative,
            "thought_process": response_thought.to_dict(),
            "coherence": response_thought.coherence_score,
            "timestamp": datetime.now().isoformat(),
            "interactive": True
        }
    
    def _parse_user_input(self, user_input: str) -> List[ConceptualPrimitive]:
        """Analitza l'entrada de l'usuari per extreure concepts."""
        
        input_lower = user_input.lower()
        concept_mapping = {
            "ordre": ConceptualPrimitive.ORDER,
            "caos": ConceptualPrimitive.CHAOS,
            "simetria": ConceptualPrimitive.SYMMETRY,
            "tensió": ConceptualPrimitive.TENSION,
            "complex": ConceptualPrimitive.COMPLEXITY,
            "simple": ConceptualPrimitive.SIMPLICITY,
            "estable": ConceptualPrimitive.STABILITY,
            "inestable": ConceptualPrimitive.INSTABILITY,
            "emergeix": ConceptualPrimitive.EMERGENCE,
            "expandeix": ConceptualPrimitive.EXPANSION,
            "contrau": ConceptualPrimitive.CONTRACTION,
            "gira": ConceptualPrimitive.ROTATION,
            "tranquil": ConceptualPrimitive.STABILITY,
            "intens": ConceptualPrimitive.TENSION,
            "bell": ConceptualPrimitive.SYMMETRY,
            "salvatge": ConceptualPrimitive.CHAOS,
        }
        
        detected_concepts = []
        for word, concept in concept_mapping.items():
            if word in input_lower:
                detected_concepts.append(concept)
        
        return detected_concepts
    
    def _generate_response_thought(self, 
                                 input_concepts: List[ConceptualPrimitive],
                                 user_input: str) -> SynestheticThought:
        """Genera un pensament de resposta."""
        
        # Basar-se en l'estat actual
        if self.engine and hasattr(self.engine, 'angular_geometry'):
            geometric_vector = self.engine.angular_geometry.state.angular_momentum.copy()
        else:
            geometric_vector = np.random.normal(0, 1, 3)
        
        # Combinar concepts d'entrada amb recuperats de memòria
        all_concepts = input_concepts.copy()
        memory_concepts = self._retrieve_semantic_concepts(geometric_vector)
        
        for concept in memory_concepts:
            if concept not in all_concepts:
                all_concepts.append(concept)
        
        # Generar pensament
        thought = SynestheticThought(
            thought_id=f"response_{datetime.now().timestamp()}",
            geometric_vector=geometric_vector,
            semantic_concepts=all_concepts,
            narrative_fragment="Pensant en resposta...",
            confidence=0.7,
            mode=ThoughtMode.SYNESTHETIC
        )
        
        return thought
    
    def _generate_interaction_narrative(self, 
                                      thought: SynestheticThought,
                                      user_input: str) -> str:
        """Genera narrativa d'interacció."""
        
        if not thought.semantic_concepts:
            return "Reflexiono sobre el teu missatge..."
        
        # Plantilles de resposta
        response_templates = [
            "Entenc el teu desig de {concepts}. La meva geometria respon amb {response}.",
            "Sento {concepts} en les teves paraules. El meu sistema es mou cap a {response}.",
            "La teva petició de {concepts} ressona amb el meu estat actual de {response}.",
            "{concepts}... Sí, percebo aquesta direcció. L'estructura fractal s'ajusta cap a {response}.",
        ]
        
        concepts_str = " i ".join([c.value for c in thought.semantic_concepts[:3]])
        
        # Determinar resposta basant-se en la coherència
        if thought.coherence_score > 0.7:
            response = "harmonia integrada"
        elif thought.coherence_score > 0.4:
            response = "tensió creativa"
        else:
            response = "exploració intuïtiva"
        
        template = np.random.choice(response_templates)
        narrative = template.format(concepts=concepts_str, response=response)
        
        return narrative
    
    def get_cortex_report(self) -> Dict[str, Any]:
        """Retorna informe del còrtex."""
        
        return {
            "cortex_statistics": self.stats,
            "memory_status": {
                "total_cells": len(self.memory_cells),
                "active_patterns": sum(1 for c in self.memory_cells.values() 
                                      if c.activation_count > 0),
                "avg_activation": np.mean([c.activation_count 
                                          for c in self.memory_cells.values()]) 
                                  if self.memory_cells else 0,
            },
            "thought_analysis": {
                "current_thoughts": len(self.current_thoughts),
                "avg_coherence": np.mean([t.coherence_score 
                                         for t in self.current_thoughts]) 
                                if self.current_thoughts else 0,
                "mode_distribution": {
                    mode.name: sum(1 for t in self.current_thoughts 
                                  if t.mode == mode)
                    for mode in ThoughtMode
                }
            },
            "configuration": self.config,
            "processing_status": self.processing
        }

# ============================================================================
# INTERFÍCIE DE TEMPS REAL
# ============================================================================

class RealTimeInterface:
    """
    Interfície asíncrona per a interacció en temps real.
    Permet interrompre l'evolució fractal mentre succeeix.
    """
    
    def __init__(self, engine_ref, cortex_ref):
        self.engine = engine_ref
        self.cortex = cortex_ref
        self.interaction_queue = asyncio.Queue()
        self.is_interacting = False
        self.interaction_lock = asyncio.Lock()
        
        # Historial d'interaccions
        self.interaction_history = []
        self.max_history = 100
        
        # Configuració
        self.config = {
            "response_timeout": 5.0,  # Segons
            "max_interruption_depth": 3,  # Interrupcions simultànies màximes
            "enable_voice_feedback": False,
            "visual_feedback": True,
        }
        
        # Task actual d'evolució
        self.current_evolution_task = None
        self.evolution_running = False
    
    async def start_interactive_session(self):
        """Inicia una sessió interactiva."""
        
        print("\n" + "="*60)
        print("🔄 SESSIÓ INTERACTIVA EN TEMPS REAL")
        print("="*60)
        print("• Escriu comandes mentre el sistema evoluciona")
        print("• 'pause' per aturar temporalment")
        print("• 'resume' per continuar")
        print("• 'state' per veure l'estat actual")
        print("• 'exit' per sortir")
        print("="*60)
        
        # Iniciar evolució en segon pla
        self.evolution_running = True
        self.current_evolution_task = asyncio.create_task(
            self._continuous_evolution()
        )
        
        # Bucle d'interacció principal
        try:
            while self.evolution_running:
                # Llegir entrada de l'usuari
                try:
                    user_input = await asyncio.wait_for(
                        self._async_input("Tu: "),
                        timeout=0.1
                    )
                    
                    if user_input:
                        await self.process_user_input(user_input)
                        
                except asyncio.TimeoutError:
                    # No hi ha entrada, continuar
                    continue
                    
        except KeyboardInterrupt:
            print("\n\nSessió interrompuda per l'usuari")
        finally:
            await self.stop_evolution()
    
    async def _async_input(self, prompt: str) -> str:
        """Entrada asíncrona."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: input(prompt))
    
    async def _continuous_evolution(self):
        """Evolució contínua en segon pla."""
        
        generation_count = 0
        
        while self.evolution_running:
            async with self.interaction_lock:
                # Verificar si estem en pausa
                if self.is_interacting:
                    await asyncio.sleep(0.5)
                    continue
                
                # Executar una generació
                try:
                    if hasattr(self.engine, 'run_generation'):
                        result = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: self.engine.run_generation(steps=2, optimize=False)
                        )
                        
                        generation_count += 1
                        
                        # Informe periòdic
                        if generation_count % 5 == 0:
                            await self._display_progress(generation_count, result)
                        
                        # Esperar breument
                        await asyncio.sleep(0.3)
                        
                except Exception as e:
                    print(f"Error en evolució: {e}")
                    await asyncio.sleep(1)
    
    async def _display_progress(self, generation: int, result: Dict[str, Any]):
        """Mostra progrés de l'evolució."""
        
        if not self.config["visual_feedback"]:
            return
        
        # Obtenir narrativa del còrtex
        if self.cortex and self.cortex.current_thoughts:
            latest_thought = self.cortex.current_thoughts[-1]
            narrative = latest_thought.narrative_fragment
        else:
            narrative = "Processant..."
        
        print(f"\n[Gen {generation}] {narrative[:60]}...")
        
        # Mostrar mètriques clau
        if result and "evolution" in result:
            final_state = result["evolution"]["final_state"]
            print(f"  Coherència: {final_state.get('coherence', 0):.3f}")
            print(f"  Entropia: {final_state.get('entropy', 0):.3f}")
    
    async def process_user_input(self, user_input: str):
        """Processa entrada de l'usuari en temps real."""
        
        input_lower = user_input.strip().lower()
        
        # Comandes de control
        if input_lower == "exit" or input_lower == "sortir":
            print("Finalitzant sessió...")
            self.evolution_running = False
            return
        
        elif input_lower == "pause" or input_lower == "atura":
            self.is_interacting = True
            print("⏸️  Evolució en pausa. Escriu 'resume' per continuar.")
            return
        
        elif input_lower == "resume" or input_lower == "continua":
            self.is_interacting = False
            print("▶️  Evolució represa.")
            return
        
        elif input_lower == "state" or input_lower == "estat":
            await self._display_current_state()
            return
        
        elif input_lower == "thoughts" or input_lower == "pensaments":
            await self._display_current_thoughts()
            return
        
        # Interacció semàntica
        async with self.interaction_lock:
            print("🤔 Pensant en la teva resposta...")
            
            # Processar a través del còrtex
            if self.cortex:
                response = self.cortex.interact(user_input)
                
                print(f"\n🧠 SVGelona_AI: {response['system_response']}")
                
                # Mostrar informació addicional
                if response['coherence'] > 0.7:
                    print(f"   (Coherència alta: {response['coherence']:.2f})")
                
                # Guardar en històric
                self.interaction_history.append(response)
                if len(self.interaction_history) > self.max_history:
                    self.interaction_history = self.interaction_history[-self.max_history:]
            
            else:
                print("⚠️  Còrtex no disponible.")
    
    async def _display_current_state(self):
        """Mostra l'estat actual del sistema."""
        
        if not self.engine:
            print("⚠️  Motor no disponible.")
            return
        
        try:
            # Obtenir estat
            state = self.engine._get_system_state_summary()
            
            print("\n" + "="*50)
            print("ESTAT ACTUAL DEL SISTEMA")
            print("="*50)
            
            print(f"Generació: {state.get('generation_count', 0)}")
            print(f"Cicatrius: {state.get('scar_archive', {}).get('total_scars', 0)}")
            
            if 'angular_geometry' in state:
                angular = state['angular_geometry']
                print(f"Fase: {angular.get('phase', 'unknown')}")
                print(f"Estabilitat: {angular.get('structural_stability', 0):.3f}")
            
            if self.cortex:
                cortex_report = self.cortex.get_cortex_report()
                print(f"Pensaments: {cortex_report['thought_analysis']['current_thoughts']}")
                print(f"Coherència mitjana: {cortex_report['thought_analysis']['avg_coherence']:.3f}")
            
            print("="*50)
            
        except Exception as e:
            print(f"Error obtenint estat: {e}")
    
    async def _display_current_thoughts(self):
        """Mostra els pensaments actuals."""
        
        if not self.cortex or not self.cortex.current_thoughts:
            print("⚠️  No hi ha pensaments actius.")
            return
        
        print("\n" + "="*50)
        print("PENSAMENTS ACTUALS")
        print("="*50)
        
        for i, thought in enumerate(self.cortex.current_thoughts[-5:]):  # Últims 5
            concepts = ", ".join([c.value for c in thought.semantic_concepts[:3]])
            print(f"{i+1}. {thought.narrative_fragment[:60]}...")
            print(f"   Concepts: {concepts}")
            print(f"   Coherència: {thought.coherence_score:.2f}")
            print(f"   Mode: {thought.mode.name}")
            print()
        
        print("="*50)
    
    async def stop_evolution(self):
        """Atura l'evolució."""
        self.evolution_running = False
        
        if self.current_evolution_task:
            self.current_evolution_task.cancel()
            try:
                await self.current_evolution_task
            except asyncio.CancelledError:
                pass
        
        print("\n✅ Evolució aturada.")
    
    def get_session_report(self) -> Dict[str, Any]:
        """Retorna informe de la sessió."""
        
        return {
            "session_statistics": {
                "interactions": len(self.interaction_history),
                "evolution_generations": getattr(self.engine, 'generation_count', 0),
                "duration_seconds": None,  # Podria calcular-se amb timestamp inicial
            },
            "recent_interactions": [
                {
                    "input": i["user_input"],
                    "response": i["system_response"][:100] + "...",
                    "coherence": i.get("coherence", 0),
                    "timestamp": i.get("timestamp", "")
                }
                for i in self.interaction_history[-5:]
            ],
            "configuration": self.config
        }

# ============================================================================
# INTEGRACIÓ AMB SVGELONA_AI
# ============================================================================

def integrate_synesthetic_core(engine_instance):
    """
    Integra el nucli sinestèsic a una instància existent de SVGelona_AI.
    
    Args:
        engine_instance: Instància de SVGelonaAI5_2
        
    Returns:
        Tupla (cortex, interface)
    """
    
    # Crear còrtex
    cortex = SynestheticCortex(engine_instance)
    
    # Iniciar pensament
    cortex.start_thinking()
    
    # Crear interfície
    interface = RealTimeInterface(engine_instance, cortex)
    
    # Integrar amb l'engine
    engine_instance.synesthetic_cortex = cortex
    engine_instance.realtime_interface = interface
    
    # Afegir mètodes nous
    def think_aloud(self, duration_seconds: float = 5.0):
        """Fa que el sistema pensi en veu alta."""
        
        if not hasattr(self, 'synesthetic_cortex'):
            print("⚠️  Còrtex sinestèsic no disponible.")
            return []
        
        print(f"\n🧠 PENSANT EN VEU ALTA ({duration_seconds}s)...")
        
        thoughts = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            if self.synesthetic_cortex.current_thoughts:
                latest = self.synesthetic_cortex.current_thoughts[-1]
                thoughts.append(latest.to_dict())
                
                print(f"  «{latest.narrative_fragment[:60]}...»")
                
                if latest.coherence_score > 0.8:
                    print(f"    ✓ Coherència alta ({latest.coherence_score:.2f})")
                
                time.sleep(1)
        
        print(f"✅ {len(thoughts)} pensaments generats.")
        return thoughts
    
    def interactive_session(self):
        """Inicia una sessió interactiva."""
        
        if not hasattr(self, 'realtime_interface'):
            print("⚠️  Interfície en temps real no disponible.")
            return
        
        try:
            # Executar en loop d'asyncio
            asyncio.run(self.realtime_interface.start_interactive_session())
        except KeyboardInterrupt:
            print("\nSessió finalitzada.")
        except Exception as e:
            print(f"Error en sessió interactiva: {e}")
    
    # Afegir mètodes a l'instància
    engine_instance.think_aloud = types.MethodType(think_aloud, engine_instance)
    engine_instance.interactive_session = types.MethodType(interactive_session, engine_instance)
    
    print("🧠 Nucli sinestèsic integrat correctament.")
    
    return cortex, interface

# ============================================================================
# SCRIPT DE DEMOSTRACIÓ
# ============================================================================

def demonstration():
    """Demostració completa del nucli sinestèsic."""
    
    print("\n" + "="*70)
    print("🧠 DEMOSTRACIÓ NUCLEI SINESTÈSIC")
    print("="*70)
    
    # Crear instància simple per a demostració
    class DemoEngine:
        def __init__(self):
            self.generation_count = 0
            self.angular_geometry = type('obj', (object,), {
                'state': type('obj', (object,), {
                    'angular_momentum': np.array([0.1, 0.5, 0.3]),
                    'torsion_tensor': np.eye(3) + np.random.normal(0, 0.1, (3, 3)),
                    'angular_entropy': 0.6,
                    'phase': 'coherence'
                })()
            })()
            self.fractal_module = type('obj', (object,), {
                'spawn_fractal_manifold': lambda seed, depth: {
                    'entropy': np.random.random(),
                    'branch_count': 2**depth
                },
                'max_depth': 8
            })()
    
    # Crear engine de demostració
    demo_engine = DemoEngine()
    
    # Crear còrtex
    cortex = SynestheticCortex(demo_engine, {
        "thought_generation_interval": 0.5,
        "max_concurrent_thoughts": 3
    })
    
    # Iniciar pensament
    cortex.start_thinking()
    
    print("1. Pensament autònom (10 segons)...")
    time.sleep(10)
    
    print("\n2. Interacció amb l'usuari...")
    responses = [
        "Crea ordre i simetria",
        "Introdueix més caos",
        "Explora complexitat",
        "Busca tensió creativa"
    ]
    
    for prompt in responses:
        print(f"\nTu: {prompt}")
        response = cortex.interact(prompt)
        print(f"IA: {response['system_response']}")
        time.sleep(2)
    
    # Aturar pensament
    cortex.stop_thinking()
    
    # Mostrar informe
    report = cortex.get_cortex_report()
    print(f"\n📊 INFORME FINAL:")
    print(f"  Pensaments generats: {report['cortex_statistics']['thoughts_generated']}")
    print(f"  Coherència mitjana: {report['thought_analysis']['avg_coherence']:.3f}")
    print(f"  Patrons de memòria: {report['memory_status']['total_cells']}")
    
    print("\n" + "="*70)
    print("✅ DEMOSTRACIÓ COMPLETADA")
    print("="*70)

if __name__ == "__main__":
    import time
    import types
    
    # Executar demostració
    demonstration()