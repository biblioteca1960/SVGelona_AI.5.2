"""
SVGelona_AI 5.2 - Sistema Axiomàtic Pont-Teoremes
Connexió entre traumes, axiomes i teoremes derivats.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
import hashlib
import json

class AxiomCategory(Enum):
    """Categories d'axiomes."""
    GEOMETRIC = "geometric"
    EVOLUTIONARY = "evolutionary"
    FRACTAL = "fractal"
    MEMORY = "memory"
    TRAUMA_RESPONSE = "trauma_response"

@dataclass
class Axiom:
    """Axioma derivat d'un trauma."""
    
    axiom_id: str
    category: AxiomCategory
    trauma_source: str  # ID del trauma que va generar l'axioma
    statement: str
    confidence: float  # 0-1, certesa de l'axioma
    derived_theorems: List[str] = field(default_factory=list)
    applications_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    last_applied: Optional[datetime] = None
    
    def __post_init__(self):
        if self.confidence < 0:
            self.confidence = 0.0
        elif self.confidence > 1:
            self.confidence = 1.0

@dataclass
class Theorem:
    """Teorema derivat d'axiomes."""
    
    theorem_id: str
    name: str
    premises: List[str]  # IDs d'axiomes
    conclusion: str
    proof_steps: List[str]
    complexity_score: float  # 0-1
    verification_status: str = "unverified"  # unverified, pending, verified, refuted
    timestamp: datetime = field(default_factory=datetime.now)

class AxiomBridgeEngine:
    """
    Motor que connecta traumes amb axiomes i deriva nous teoremes.
    """
    
    def __init__(self, scar_archive):
        self.scar_archive = scar_archive
        
        # Bases de coneixement
        self.axioms: Dict[str, Axiom] = {}
        self.theorems: Dict[str, Theorem] = {}
        
        # Relacions entre traumes i axiomes
        self.trauma_to_axioms: Dict[str, List[str]] = {}
        
        # Contradiccions i consistència
        self.contradictions: List[Tuple[str, str]] = []  # Pairs d'axiomes contradictoris
        
        # Configuració
        self.config = {
            "min_confidence_threshold": 0.6,
            "max_axioms_per_trauma": 3,
            "theorem_derivation_depth": 3,
            "consistency_check_interval": 50,
            "axiom_aging_factor": 0.99  # Els axiomes perden confiança amb el temps
        }
        
        # Estadístiques
        self.stats = {
            "axioms_generated": 0,
            "theorems_derived": 0,
            "contradictions_found": 0,
            "consistency_checks": 0
        }
        
        # Inicialitzar amb axiomes bàsics
        self._initialize_basic_axioms()
    
    def _initialize_basic_axioms(self):
        """Inicialitza amb un conjunt bàsic d'axiomes."""
        basic_axioms = [
            {
                "category": AxiomCategory.GEOMETRIC,
                "statement": "El sistema tendeix a mantenir la coherència euclidiana",
                "confidence": 0.95
            },
            {
                "category": AxiomCategory.EVOLUTIONARY,
                "statement": "Els traumes generen adaptacions que milloren la resiliència",
                "confidence": 0.85
            },
            {
                "category": AxiomCategory.FRACTAL,
                "statement": "La complexitat fractal sorgeix de regles simples iteratives",
                "confidence": 0.9
            },
            {
                "category": AxiomCategory.MEMORY,
                "statement": "La memòria d'accés freqüent té major prioritat de conservació",
                "confidence": 0.8
            }
        ]
        
        for axiom_data in basic_axioms:
            axiom_id = hashlib.sha256(
                f"basic_{axiom_data['statement']}".encode()
            ).hexdigest()[:16]
            
            axiom = Axiom(
                axiom_id=axiom_id,
                category=axiom_data["category"],
                trauma_source="system_init",
                statement=axiom_data["statement"],
                confidence=axiom_data["confidence"]
            )
            
            self.axioms[axiom_id] = axiom
        
        self.stats["axioms_generated"] = len(basic_axioms)
    
    def process_trauma(self, trauma_type: str, trauma_data: Dict[str, Any]) -> List[Axiom]:
        """
        Processa un trauma i deriva nous axiomes.
        
        Args:
            trauma_type: Tipus de trauma (ex: "boundary_collision")
            trauma_data: Dades associades al trauma
            
        Returns:
            Llista d'axiomes derivats
        """
        derived_axioms = []
        
        # Generar ID únic per al trauma
        trauma_hash = hashlib.sha256(
            f"{trauma_type}_{json.dumps(trauma_data, sort_keys=True)}".encode()
        ).hexdigest()[:16]
        
        # Derivar axiomes basats en el tipus de trauma
        if trauma_type == "boundary_collision":
            axioms = self._derive_boundary_axioms(trauma_data, trauma_hash)
            derived_axioms.extend(axioms)
        
        elif trauma_type == "memory_overflow":
            axioms = self._derive_memory_axioms(trauma_data, trauma_hash)
            derived_axioms.extend(axioms)
        
        elif trauma_type == "fractal_collapse":
            axioms = self._derive_fractal_axioms(trauma_data, trauma_hash)
            derived_axioms.extend(axioms)
        
        elif trauma_type == "optimized_evolution_error":
            axioms = self._derive_error_axioms(trauma_data, trauma_hash)
            derived_axioms.extend(axioms)
        
        # Limitar nombre d'axiomes derivats
        max_axioms = self.config["max_axioms_per_trauma"]
        if len(derived_axioms) > max_axioms:
            # Ordenar per confiança i agafar els millors
            derived_axioms.sort(key=lambda a: a.confidence, reverse=True)
            derived_axioms = derived_axioms[:max_axioms]
        
        # Guardar axiomes derivats
        for axiom in derived_axioms:
            self.axioms[axiom.axiom_id] = axiom
            
            # Registrar relació trauma-axioma
            if trauma_hash not in self.trauma_to_axioms:
                self.trauma_to_axioms[trauma_hash] = []
            self.trauma_to_axioms[trauma_hash].append(axiom.axiom_id)
        
        self.stats["axioms_generated"] += len(derived_axioms)
        
        # Verificar consistència periòdicament
        if self.stats["axioms_generated"] % self.config["consistency_check_interval"] == 0:
            self.check_axiom_consistency()
        
        return derived_axioms
    
    def _derive_boundary_axioms(self, trauma_data: Dict[str, Any], trauma_hash: str) -> List[Axiom]:
        """Deriva axiomes d'una col·lisió amb límits."""
        axioms = []
        
        position = np.array(trauma_data.get("position", [0, 0, 0]))
        distance = trauma_data.get("distance", 0.0)
        
        # Axioma 1: Els límits defineixen l'espai de possibilitats
        axiom1_statement = f"Els límits a distància {distance:.2f} defineixen l'espai evolutiu"
        axiom1_id = hashlib.sha256(
            f"boundary_{trauma_hash}_1".encode()
        ).hexdigest()[:16]
        
        axiom1 = Axiom(
            axiom_id=axiom1_id,
            category=AxiomCategory.GEOMETRIC,
            trauma_source=trauma_hash,
            statement=axiom1_statement,
            confidence=0.7 + 0.2 * (1.0 - min(1.0, distance / 10.0))  # Més confiança a límits propers
        )
        axioms.append(axiom1)
        
        # Axioma 2: L'evolució tendeix a explorar límits
        axiom2_statement = "El sistema evolucionari explora activament els límits del seu espai"
        axiom2_id = hashlib.sha256(
            f"boundary_{trauma_hash}_2".encode()
        ).hexdigest()[:16]
        
        axiom2 = Axiom(
            axiom_id=axiom2_id,
            category=AxiomCategory.EVOLUTIONARY,
            trauma_source=trauma_hash,
            statement=axiom2_statement,
            confidence=0.65
        )
        axioms.append(axiom2)
        
        return axioms
    
    def _derive_memory_axioms(self, trauma_data: Dict[str, Any], trauma_hash: str) -> List[Axiom]:
        """Deriva axiomes d'un desbordament de memòria."""
        axioms = []
        
        memory_used = trauma_data.get("memory_used_mb", 0)
        memory_limit = trauma_data.get("memory_limit_mb", 100)
        
        # Axioma 1: La memòria és un recurs limitat que cal gestionar
        utilization = memory_used / memory_limit if memory_limit > 0 else 1.0
        axiom1_statement = f"La utilització de memòria del {utilization:.0%} requereix gestió activa"
        axiom1_id = hashlib.sha256(
            f"memory_{trauma_hash}_1".encode()
        ).hexdigest()[:16]
        
        axiom1 = Axiom(
            axiom_id=axiom1_id,
            category=AxiomCategory.MEMORY,
            trauma_source=trauma_hash,
            statement=axiom1_statement,
            confidence=0.8
        )
        axioms.append(axiom1)
        
        # Axioma 2: La poda selectiva millora el rendiment
        axiom2_statement = "L'eliminació selectiva de memòria poc útil millora l'eficàcia del sistema"
        axiom2_id = hashlib.sha256(
            f"memory_{trauma_hash}_2".encode()
        ).hexdigest()[:16]
        
        axiom2 = Axiom(
            axiom_id=axiom2_id,
            category=AxiomCategory.MEMORY,
            trauma_source=trauma_hash,
            statement=axiom2_statement,
            confidence=0.75
        )
        axioms.append(axiom2)
        
        return axioms
    
    def _derive_fractal_axioms(self, trauma_data: Dict[str, Any], trauma_hash: str) -> List[Axiom]:
        """Deriva axiomes d'un col·lapse fractal."""
        axioms = []
        
        depth = trauma_data.get("depth", 5)
        complexity = trauma_data.get("complexity", 100)
        
        # Axioma 1: La complexitat fractal té límits pràctics
        axiom1_statement = f"La profunditat fractal de {depth} nivells pot apropar-se als límits de complexitat"
        axiom1_id = hashlib.sha256(
            f"fractal_{trauma_hash}_1".encode()
        ).hexdigest()[:16]
        
        axiom1 = Axiom(
            axiom_id=axiom1_id,
            category=AxiomCategory.FRACTAL,
            trauma_source=trauma_hash,
            statement=axiom1_statement,
            confidence=0.7
        )
        axioms.append(axiom1)
        
        # Axioma 2: L'adaptabilitat requereix ajustos de complexitat
        axiom2_statement = "Els sistemes fractals han de balancejar complexitat i estabilitat"
        axiom2_id = hashlib.sha256(
            f"fractal_{trauma_hash}_2".encode()
        ).hexdigest()[:16]
        
        axiom2 = Axiom(
            axiom_id=axiom2_id,
            category=AxiomCategory.EVOLUTIONARY,
            trauma_source=trauma_hash,
            statement=axiom2_statement,
            confidence=0.68
        )
        axioms.append(axiom2)
        
        return axioms
    
    def _derive_error_axioms(self, trauma_data: Dict[str, Any], trauma_hash: str) -> List[Axiom]:
        """Deriva axiomes d'un error d'evolució optimitzada."""
        axioms = []
        
        error_msg = trauma_data.get("error", "unknown")
        
        # Axioma 1: Els errors revelen límits dels algorismes
        axiom1_statement = f"L'error '{error_msg[:50]}...' indica un límit en l'algorisme actual"
        axiom1_id = hashlib.sha256(
            f"error_{trauma_hash}_1".encode()
        ).hexdigest()[:16]
        
        axiom1 = Axiom(
            axiom_id=axiom1_id,
            category=AxiomCategory.EVOLUTIONARY,
            trauma_source=trauma_hash,
            statement=axiom1_statement,
            confidence=0.6
        )
        axioms.append(axiom1)
        
        # Axioma 2: La robustesa requereix maneig d'errors
        axiom2_statement = "Els sistemes robustes incorporen mecanismes de recuperació d'errors"
        axiom2_id = hashlib.sha256(
            f"error_{trauma_hash}_2".encode()
        ).hexdigest()[:16]
        
        axiom2 = Axiom(
            axiom_id=axiom2_id,
            category=AxiomCategory.TRAUMA_RESPONSE,
            trauma_source=trauma_hash,
            statement=axiom2_statement,
            confidence=0.72
        )
        axioms.append(axiom2)
        
        return axioms
    
    def derive_theorems(self, max_theorems: int = 10) -> List[Theorem]:
        """
        Deriva nous teoremes a partir dels axiomes existents.
        
        Args:
            max_theorems: Nombre màxim de teoremes a derivar
            
        Returns:
            Llista de teoremes derivats
        """
        derived_theorems = []
        
        # Agrupar axiomes per categoria
        axioms_by_category = {}
        for axiom in self.axioms.values():
            category = axiom.category.value
            if category not in axioms_by_category:
                axioms_by_category[category] = []
            axioms_by_category[category].append(axiom)
        
        # Derivar teoremes creuant categories
        categories = list(axioms_by_category.keys())
        
        for i in range(min(max_theorems, len(categories) * 2)):
            # Seleccionar categories per a creuar
            if len(categories) >= 2:
                cat1, cat2 = np.random.choice(categories, 2, replace=False)
                
                if (cat1 in axioms_by_category and axioms_by_category[cat1] and
                    cat2 in axioms_by_category and axioms_by_category[cat2]):
                    
                    # Seleccionar axiomes
                    axiom1 = np.random.choice(axioms_by_category[cat1])
                    axiom2 = np.random.choice(axioms_by_category[cat2])
                    
                    # Derivar teorema
                    theorem = self._derive_theorem_from_axioms(axiom1, axiom2)
                    
                    if theorem:
                        self.theorems[theorem.theorem_id] = theorem
                        
                        # Actualitzar referències
                        axiom1.derived_theorems.append(theorem.theorem_id)
                        axiom2.derived_theorems.append(theorem.theorem_id)
                        
                        derived_theorems.append(theorem)
                        self.stats["theorems_derived"] += 1
        
        return derived_theorems
    
    def _derive_theorem_from_axioms(self, axiom1: Axiom, axiom2: Axiom) -> Optional[Theorem]:
        """Deriva un teorema a partir de dos axiomes."""
        
        # Evitar derivar del mateix axioma
        if axiom1.axiom_id == axiom2.axiom_id:
            return None
        
        # Crear ID únic
        theorem_id = hashlib.sha256(
            f"theorem_{axiom1.axiom_id}_{axiom2.axiom_id}".encode()
        ).hexdigest()[:16]
        
        # Generar nom descriptiu
        theorem_name = f"Teorema_{axiom1.category.value[:3]}_{axiom2.category.value[:3]}_{theorem_id[:6]}"
        
        # Generar conclusió basada en la combinació d'axiomes
        conclusion = self._generate_conclusion(axiom1, axiom2)
        
        # Generar passos de prova
        proof_steps = [
            f"Premisa 1: {axiom1.statement} (confiança: {axiom1.confidence:.2f})",
            f"Premisa 2: {axiom2.statement} (confiança: {axiom2.confidence:.2f})",
            f"Deducció: Si ambdues premisses són certes, llavors {conclusion}"
        ]
        
        # Calcular puntuació de complexitat
        complexity = (axiom1.confidence + axiom2.confidence) / 2
        
        theorem = Theorem(
            theorem_id=theorem_id,
            name=theorem_name,
            premises=[axiom1.axiom_id, axiom2.axiom_id],
            conclusion=conclusion,
            proof_steps=proof_steps,
            complexity_score=complexity,
            verification_status="unverified"
        )
        
        return theorem
    
    def _generate_conclusion(self, axiom1: Axiom, axiom2: Axiom) -> str:
        """Genera una conclusió a partir de dos axiomes."""
        
        # Patrons de conclusió basats en combinacions de categories
        category_pair = (axiom1.category, axiom2.category)
        
        conclusion_templates = {
            (AxiomCategory.GEOMETRIC, AxiomCategory.EVOLUTIONARY): 
                "Les estructures geomètriques guien el procés evolutiu del sistema",
            
            (AxiomCategory.FRACTAL, AxiomCategory.MEMORY):
                "La memòria fractal optimitza l'emmagatzematge de patrons complexos",
            
            (AxiomCategory.EVOLUTIONARY, AxiomCategory.TRAUMA_RESPONSE):
                "La resposta als traumes impulsa l'adaptació evolutiva",
            
            (AxiomCategory.GEOMETRIC, AxiomCategory.FRACTAL):
                "La geometria subjau a l'emergència de patrons fractals"
        }
        
        # Cercar patró específic
        if category_pair in conclusion_templates:
            return conclusion_templates[category_pair]
        elif (category_pair[1], category_pair[0]) in conclusion_templates:
            return conclusion_templates[(category_pair[1], category_pair[0])]
        
        # Patró general
        return (f"La interacció entre {axiom1.category.value} i {axiom2.category.value} "
                f"genera noves propietats emergents")
    
    def check_axiom_consistency(self) -> Dict[str, Any]:
        """
        Verifica la consistència dels axiomes i identifica contradiccions.
        
        Returns:
            Informe de consistència
        """
        self.stats["consistency_checks"] += 1
        
        contradictions_found = []
        consistent_pairs = []
        
        axioms_list = list(self.axioms.values())
        
        for i in range(len(axioms_list)):
            for j in range(i + 1, len(axioms_list)):
                axiom_i = axioms_list[i]
                axiom_j = axioms_list[j]
                
                # Verificar contradicció (simplificat)
                is_contradiction = self._check_contradiction(axiom_i, axiom_j)
                
                if is_contradiction:
                    contradictions_found.append((axiom_i.axiom_id, axiom_j.axiom_id))
                    self.contradictions.append((axiom_i.axiom_id, axiom_j.axiom_id))
                else:
                    consistent_pairs.append((axiom_i.axiom_id, axiom_j.axiom_id))
        
        self.stats["contradictions_found"] += len(contradictions_found)
        
        return {
            "total_axioms": len(self.axioms),
            "contradictions_found": len(contradictions_found),
            "consistent_pairs": len(consistent_pairs),
            "consistency_score": len(consistent_pairs) / max(1, len(consistent_pairs) + len(contradictions_found)),
            "contradiction_examples": contradictions_found[:5]  # Limitar per a l'informe
        }
    
    def _check_contradiction(self, axiom1: Axiom, axiom2: Axiom) -> bool:
        """Verifica si dos axiomes es contradiuen (simplificat)."""
        
        # Contradiccions bàsiques basades en paraules clau
        negation_keywords = ["no ", "mai ", "cap ", "contra ", "evitar ", "no pot ", "impossible "]
        
        statement1 = axiom1.statement.lower()
        statement2 = axiom2.statement.lower()
        
        # Verificar si un afirma el que l'altre nega
        for neg_keyword in negation_keywords:
            if (neg_keyword in statement1 and 
                neg_keyword not in statement2 and
                any(word in statement2 for word in statement1.split() if len(word) > 3)):
                return True
        
        # Verificar categories contradictòries (simplificat)
        if (axiom1.category == AxiomCategory.GEOMETRIC and 
            axiom2.category == AxiomCategory.FRACTAL and
            "límit" in statement1 and "infinit" in statement2):
            return True
        
        return False
    
    def get_axiom_system_report(self) -> Dict[str, Any]:
        """Genera informe complet del sistema axiomàtic."""
        
        consistency_report = self.check_axiom_consistency()
        
        # Distribució per categories
        category_dist = {}
        for axiom in self.axioms.values():
            cat = axiom.category.value
            category_dist[cat] = category_dist.get(cat, 0) + 1
        
        # Confiança mitjana
        confidences = [a.confidence for a in self.axioms.values()]
        
        return {
            "system_stats": self.stats,
            "consistency": consistency_report,
            "category_distribution": category_dist,
            "confidence_metrics": {
                "avg_confidence": np.mean(confidences) if confidences else 0,
                "min_confidence": min(confidences) if confidences else 0,
                "max_confidence": max(confidences) if confidences else 0,
                "high_confidence_axioms": sum(1 for c in confidences if c > 0.8)
            },
            "theorem_system": {
                "total_theorems": len(self.theorems),
                "verified_theorems": sum(1 for t in self.theorems.values() 
                                        if t.verification_status == "verified"),
                "avg_complexity": np.mean([t.complexity_score for t in self.theorems.values()]) 
                                if self.theorems else 0
            }
        }
    
    def apply_axiom(self, axiom_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplica un axioma a un context específic.
        
        Args:
            axiom_id: ID de l'axioma a aplicar
            context: Context d'aplicació
            
        Returns:
            Resultat de l'aplicació
        """
        if axiom_id not in self.axioms:
            return {"success": False, "error": "Axioma no trobat"}
        
        axiom = self.axioms[axiom_id]
        
        # Actualitzar estadístiques de l'axioma
        axiom.applications_count += 1
        axiom.last_applied = datetime.now()
        
        # Aplicar basant-se en la categoria (simplificat)
        application_result = self._apply_axiom_by_category(axiom, context)
        
        return {
            "success": True,
            "axiom_applied": axiom.statement,
            "confidence": axiom.confidence,
            "result": application_result,
            "applications_count": axiom.applications_count
        }
    
    def _apply_axiom_by_category(self, axiom: Axiom, context: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica un axioma basant-se en la seva categoria."""
        
        if axiom.category == AxiomCategory.GEOMETRIC:
            return self._apply_geometric_axiom(axiom, context)
        
        elif axiom.category == AxiomCategory.EVOLUTIONARY:
            return self._apply_evolutionary_axiom(axiom, context)
        
        elif axiom.category == AxiomCategory.FRACTAL:
            return self._apply_fractal_axiom(axiom, context)
        
        elif axiom.category == AxiomCategory.MEMORY:
            return self._apply_memory_axiom(axiom, context)
        
        else:
            return {"effect": "general", "impact": 0.5 * axiom.confidence}
    
    def _apply_geometric_axiom(self, axiom: Axiom, context: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica un axioma geomètric."""
        position = context.get("position", np.zeros(3))
        
        # Efecte simplificat: ajustar direcció basant-se en l'axioma
        adjustment = np.array([axiom.confidence * 0.1, 0, 0])
        
        return {
            "effect": "geometric_adjustment",
            "adjustment_vector": adjustment.tolist(),
            "magnitude": float(np.linalg.norm(adjustment))
        }
    
    def _apply_evolutionary_axiom(self, axiom: Axiom, context: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica un axioma evolutiu."""
        evolutionary_pressure = context.get("evolutionary_pressure", 0.5)
        
        # Augmentar pressió evolutiva
        new_pressure = min(1.0, evolutionary_pressure + 0.1 * axiom.confidence)
        
        return {
            "effect": "evolutionary_pressure_increase",
            "pressure_before": evolutionary_pressure,
            "pressure_after": new_pressure,
            "increase": new_pressure - evolutionary_pressure
        }
    
    def _apply_fractal_axiom(self, axiom: Axiom, context: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica un axioma fractal."""
        fractal_complexity = context.get("fractal_complexity", 10)
        
        # Ajustar complexitat
        complexity_factor = 1.0 + 0.2 * axiom.confidence
        new_complexity = fractal_complexity * complexity_factor
        
        return {
            "effect": "fractal_complexity_adjustment",
            "complexity_before": fractal_complexity,
            "complexity_after": new_complexity,
            "factor": complexity_factor
        }
    
    def _apply_memory_axiom(self, axiom: Axiom, context: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica un axioma de memòria."""
        memory_utilization = context.get("memory_utilization", 0.5)
        
        # Millorar eficiència de memòria
        efficiency_gain = 0.15 * axiom.confidence
        new_utilization = memory_utilization * (1.0 - efficiency_gain)
        
        return {
            "effect": "memory_efficiency_gain",
            "utilization_before": memory_utilization,
            "utilization_after": new_utilization,
            "efficiency_gain": efficiency_gain
        }