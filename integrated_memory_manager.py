"""
SVGelona_AI 5.2 - Gestor de Memòria Integrat
Integra poda selectiva, cerca optimitzada i consolidació.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
import heapq
from dataclasses import dataclass, field
from enum import Enum
import json

class MemoryPriority(Enum):
    """Prioritat de memòria per a gestió de recursos."""
    CRITICAL = 4     # Memòria crítica (no es pot eliminar)
    HIGH = 3         # Alta prioritat (axiomes, patrons arquetípics)
    MEDIUM = 2       # Prioritat mitjana (cicatrius freqüents)
    LOW = 1          # Baixa prioritat (cicatrius antigues/poc utilitzades)
    EPHEMERAL = 0    # Memòria efímera (pot eliminar-se)

@dataclass(order=True)
class MemoryItem:
    """Item de memòria amb prioritat i mètriques."""
    
    item_id: str
    priority: MemoryPriority
    last_accessed: datetime
    access_count: int
    memory_footprint: int  # En bytes estimats
    utility_score: float   # 0-1, utilitat estimada
    
    # Per a ordenació per prioritat i utilitat
    sort_key: Tuple[int, float, datetime] = field(init=False)
    
    def __post_init__(self):
        """Calcula clau d'ordenació."""
        # Invertir prioritat (més alta = número més baix)
        priority_val = 4 - self.priority.value
        
        # Invertir data (més recent = número més alt)
        # Convertir a timestamp per a comparació
        timestamp_val = -self.last_accessed.timestamp()
        
        self.sort_key = (priority_val, -self.utility_score, timestamp_val)
    
    def should_evict(self, 
                    memory_pressure: float,
                    min_utility_threshold: float = 0.2) -> bool:
        """Determina si aquest item hauria de ser eliminat."""
        if self.priority in [MemoryPriority.CRITICAL, MemoryPriority.HIGH]:
            return False
        
        # Ajustar llindar basant-se en pressió de memòria
        adjusted_threshold = min_utility_threshold * (1.0 + memory_pressure)
        
        if self.utility_score < adjusted_threshold:
            return True
        
        # Items molt antics i poc utilitzats
        days_since_access = (datetime.now() - self.last_accessed).days
        if days_since_access > 30 and self.access_count < 5:
            return True
        
        return False

class IntegratedMemoryManager:
    """
    Gestor de memòria que integra totes les optimitzacions:
    
    1. Indexació espacial per a cerca ràpida
    2. Poda selectiva basada en utilitat
    3. Consolidació de patrons
    4. Gestió de prioritat
    5. Compressió de memòria
    """
    
    def __init__(self, scar_archive, axiom_bridge):
        self.scar_archive = scar_archive
        self.axiom_bridge = axiom_bridge
        
        # Sistema de priorització
        self.memory_items: Dict[str, MemoryItem] = {}
        self.priority_queue: List[Tuple[Tuple, str]] = []  # Min-heap per (sort_key, item_id)
        
        # Configuració
        self.config = {
            "max_memory_mb": 100,                # Límit de memòria
            "target_memory_utilization": 0.7,    # Utilització objectiu
            "eviction_batch_size": 10,           # Quants eliminar per lot
            "consolidation_interval": 100,       # Cicles entre consolidacions
            "min_utility_threshold": 0.2,        # Llindar mínim d'utilitat
            "critical_items": set()              # Items que mai s'eliminen
        }
        
        # Estadístiques
        self.stats = {
            "total_items": 0,
            "memory_used_mb": 0,
            "items_evicted": 0,
            "consolidations_performed": 0,
            "compression_ratio": 1.0,
            "cache_hit_rate": 0.0
        }
        
        # Inicialitzar
        self._initialize_memory_items()
    
    def _initialize_memory_items(self):
        """Inicialitza items de memòria per a cicatrius existents."""
        for scar_id, scar in self.scar_archive.scars.items():
            self._add_scar_to_memory(scar)
    
    def _add_scar_to_memory(self, scar):
        """Afegeix una cicatriu al sistema de gestió de memòria."""
        # Calcular prioritat
        priority = self._calculate_scar_priority(scar)
        
        # Calcular utilitat
        utility = self._calculate_scar_utility(scar)
        
        # Estimar empremta de memòria
        memory_footprint = self._estimate_memory_footprint(scar)
        
        item = MemoryItem(
            item_id=scar.scar_id,
            priority=priority,
            last_accessed=scar.last_accessed,
            access_count=scar.access_count,
            memory_footprint=memory_footprint,
            utility_score=utility
        )
        
        self.memory_items[scar.scar_id] = item
        heapq.heappush(self.priority_queue, (item.sort_key, scar.scar_id))
        
        # Actualitzar estadístiques
        self.stats["total_items"] += 1
        self.stats["memory_used_mb"] += memory_footprint / (1024 * 1024)
    
    def _calculate_scar_priority(self, scar) -> MemoryPriority:
        """Calcula prioritat d'una cicatriu."""
        # Verificar si és crítica
        if scar.scar_id in self.config["critical_items"]:
            return MemoryPriority.CRITICAL
        
        # Factors per a prioritat alta:
        # 1. Referenciada per axiomes
        axiom_references = 0
        for axiom in self.axiom_bridge.axioms.values():
            if scar.scar_id[:8] in str(axiom.trauma_source):
                axiom_references += 1
        
        if axiom_references > 0:
            return MemoryPriority.HIGH
        
        # 2. Molt utilitzada recentment
        hours_since_access = (datetime.now() - scar.last_accessed).total_seconds() / 3600
        if hours_since_access < 1 and scar.access_count > 10:
            return MemoryPriority.HIGH
        
        # 3. Alt potencial evolutiu
        if scar.evolutionary_potential > 0.8:
            return MemoryPriority.HIGH
        
        # 4. Ús moderat
        if scar.access_count > 5:
            return MemoryPriority.MEDIUM
        
        # 5. Antiga i poc utilitzada
        days_since_creation = (datetime.now() - scar.timestamp).days
        if days_since_creation > 7 and scar.access_count < 3:
            return MemoryPriority.LOW
        
        # Per defecte: efímera
        return MemoryPriority.EPHEMERAL
    
    def _calculate_scar_utility(self, scar) -> float:
        """Calcula puntuació d'utilitat d'una cicatriu (0-1)."""
        factors = []
        
        # Factor 1: Ús recent (40%)
        hours_since_access = (datetime.now() - scar.last_accessed).total_seconds() / 3600
        recency_factor = 1.0 / (1.0 + hours_since_access / 24.0)  # Decau en 24h
        factors.append(recency_factor * 0.4)
        
        # Factor 2: Freqüència d'accés (25%)
        access_factor = min(1.0, scar.access_count / 20.0)  # Normalitzat a 20 accessos
        factors.append(access_factor * 0.25)
        
        # Factor 3: Potencial evolutiu (20%)
        factors.append(scar.evolutionary_potential * 0.2)
        
        # Factor 4: Edat (menys important, 15%)
        days_since_creation = (datetime.now() - scar.timestamp).days
        age_factor = 1.0 / (1.0 + days_since_creation / 30.0)  # Decau en 30 dies
        factors.append(age_factor * 0.15)
        
        return sum(factors)
    
    def _estimate_memory_footprint(self, scar) -> int:
        """Estima empremta de memòria d'una cicatriu en bytes."""
        footprint = 0
        
        # ID: ~16 bytes
        footprint += len(scar.scar_id.encode('utf-8'))
        
        # Posició: 3 * 4 bytes = 12 bytes (float32)
        footprint += 12
        
        # Timestamps: 2 * 8 bytes = 16 bytes
        footprint += 16
        
        # Lliçons: variable
        for lesson in scar.lessons_learned:
            footprint += len(lesson.encode('utf-8'))
        
        # Altres atributs
        footprint += 20  # Potencial, access_count, etc.
        
        return footprint
    
    def check_memory_pressure(self) -> float:
        """Calcula pressió de memòria actual (0-1)."""
        memory_used = self.stats["memory_used_mb"]
        memory_limit = self.config["max_memory_mb"]
        
        if memory_limit <= 0:
            return 0.0
        
        pressure = memory_used / memory_limit
        
        # Si estem per sobre del límit, pressió > 1
        return pressure
    
    def perform_memory_management(self) -> Dict[str, Any]:
        """
        Executa gestió completa de memòria.
        
        Inclou:
        1. Verificació de pressió
        2. Evicció d'items de baixa utilitat
        3. Consolidació de memòria
        4. Reorganització
        """
        management_report = {
            "timestamp": datetime.now().isoformat(),
            "memory_pressure": self.check_memory_pressure(),
            "actions_taken": [],
            "items_affected": {},
            "performance_impact": {}
        }
        
        # PAS 1: Verificar si cal prendre acció
        memory_pressure = self.check_memory_pressure()
        
        if memory_pressure < self.config["target_memory_utilization"]:
            management_report["actions_taken"].append("no_action_required")
            return management_report
        
        # PAS 2: Evicció d'items de baixa utilitat
        items_to_evict = []
        
        # Recollir items candidats per a evicció
        for item_id, item in self.memory_items.items():
            if item.should_evict(memory_pressure, self.config["min_utility_threshold"]):
                items_to_evict.append((item.sort_key, item_id))
        
        # Ordenar per prioritat (més baixa primer)
        items_to_evict.sort()
        
        # Limitar nombre d'eviccions
        max_evictions = min(self.config["eviction_batch_size"], len(items_to_evict))
        items_to_evict = items_to_evict[:max_evictions]
        
        # Aplicar eviccions
        evicted_scars = []
        evicted_memory = 0
        
        for _, item_id in items_to_evict:
            if self._evict_item(item_id):
                evicted_scars.append(item_id[:8])
                
                if item_id in self.memory_items:
                    evicted_memory += self.memory_items[item_id].memory_footprint
                    del self.memory_items[item_id]
                
                self.stats["items_evicted"] += 1
        
        if evicted_scars:
            management_report["actions_taken"].append("memory_eviction")
            management_report["items_affected"]["evicted_scars"] = evicted_scars
            management_report["items_affected"]["evicted_count"] = len(evicted_scars)
            management_report["performance_impact"]["memory_freed_mb"] = (
                evicted_memory / (1024 * 1024)
            )
            
            # Actualitzar estadístiques
            self.stats["memory_used_mb"] -= evicted_memory / (1024 * 1024)
            self.stats["total_items"] -= len(evicted_scars)
        
        # PAS 3: Consolidació de memòria (periòdica)
        if self.stats["total_items"] % self.config["consolidation_interval"] == 0:
            consolidation_result = self._perform_memory_consolidation()
            
            if consolidation_result:
                management_report["actions_taken"].append("memory_consolidation")
                management_report.update(consolidation_result)
                self.stats["consolidations_performed"] += 1
        
        # PAS 4: Reorganització de prioritat (si cal)
        if memory_pressure > 0.8:
            self._reprioritize_items()
            management_report["actions_taken"].append("reprioritization")
        
        # PAS 5: Neteja del heap de prioritat
        self._cleanup_priority_queue()
        
        # Actualitzar pressió després de les accions
        management_report["final_memory_pressure"] = self.check_memory_pressure()
        management_report["memory_stats"] = {
            "items_total": self.stats["total_items"],
            "memory_used_mb": self.stats["memory_used_mb"],
            "memory_limit_mb": self.config["max_memory_mb"]
        }
        
        return management_report
    
    def _evict_item(self, item_id: str) -> bool:
        """Elimina un item de memòria."""
        # Verificar que no sigui crític
        if item_id in self.config["critical_items"]:
            return False
        
        # Eliminar de l'arxiu de cicatrius
        success = self.scar_archive.remove_scar(item_id)
        
        return success
    
    def _perform_memory_consolidation(self) -> Dict[str, Any]:
        """Consolida memòria mitjançant tècniques de compressió."""
        consolidation_report = {
            "start_time": datetime.now().isoformat(),
            "compression_applied": False,
            "memory_saved_mb": 0,
            "patterns_consolidated": 0
        }
        
        # Tècnica 1: Compressió de lliçons redundants
        lesson_patterns = self._find_redundant_lessons()
        
        if lesson_patterns:
            memory_saved = self._compress_redundant_lessons(lesson_patterns)
            consolidation_report["compression_applied"] = True
            consolidation_report["memory_saved_mb"] += memory_saved / (1024 * 1024)
            consolidation_report["patterns_consolidated"] += len(lesson_patterns)
        
        # Tècnica 2: Agrupació de cicatrius similars
        similar_scar_groups = self._find_similar_scars()
        
        for group in similar_scar_groups:
            if len(group) > 2:
                memory_saved = self._consolidate_similar_scars(group)
                consolidation_report["memory_saved_mb"] += memory_saved / (1024 * 1024)
                consolidation_report["patterns_consolidated"] += 1
        
        consolidation_report["end_time"] = datetime.now().isoformat()
        
        return consolidation_report
    
    def _find_redundant_lessons(self) -> List[Set[str]]:
        """Troba lliçons redundants entre cicatrius."""
        # Indexar lliçons per contingut
        lesson_index = defaultdict(set)
        
        for scar_id, scar in self.scar_archive.scars.items():
            for lesson in scar.lessons_learned:
                # Normalitzar lliçó (minúscules, sense espais extra)
                normalized = lesson.lower().strip()
                lesson_index[normalized].add(scar_id)
        
        # Trobar lliçons que apareixen en múltiples cicatrius
        redundant_sets = []
        
        for lesson, scar_ids in lesson_index.items():
            if len(scar_ids) > 1:
                redundant_sets.append(scar_ids)
        
        return redundant_sets
    
    def _compress_redundant_lessons(self, lesson_patterns: List[Set[str]]) -> int:
        """Comprimeix lliçons redundants."""
        memory_saved = 0
        
        for scar_ids in lesson_patterns:
            # Trobar lliçons comunes
            common_lessons = set()
            first = True
            
            for scar_id in scar_ids:
                if scar_id in self.scar_archive.scars:
                    scar = self.scar_archive.scars[scar_id]
                    if first:
                        common_lessons = set(scar.lessons_learned)
                        first = False
                    else:
                        common_lessons.intersection_update(scar.lessons_learned)
            
            # Si hi ha lliçons comunes, es poden comprimir
            if common_lessons:
                # Crear referència compartida
                reference_id = f"shared_lessons_{hash(tuple(sorted(common_lessons)))}"
                
                # Estimar memòria estalviada
                bytes_per_lesson = sum(len(l.encode('utf-8')) for l in common_lessons)
                memory_saved += bytes_per_lesson * (len(scar_ids) - 1)
        
        return memory_saved
    
    def _find_similar_scars(self, similarity_threshold: float = 0.7) -> List[List[str]]:
        """Troba grups de cicatrius similars."""
        scars = list(self.scar_archive.scars.items())
        groups = []
        processed = set()
        
        for i, (id1, scar1) in enumerate(scars):
            if id1 in processed:
                continue
            
            similar_group = [id1]
            
            for j, (id2, scar2) in enumerate(scars):
                if i == j or id2 in processed:
                    continue
                
                similarity = self._calculate_scar_similarity(scar1, scar2)
                if similarity > similarity_threshold:
                    similar_group.append(id2)
            
            if len(similar_group) > 1:
                groups.append(similar_group)
                processed.update(similar_group)
        
        return groups
    
    def _calculate_scar_similarity(self, scar1, scar2) -> float:
        """Calcula similitud entre dues cicatrius."""
        factors = []
        
        # 1. Similitud de posició (25%)
        pos_distance = np.linalg.norm(scar1.position - scar2.position)
        pos_similarity = np.exp(-pos_distance / self.scar_archive.cell_size)
        factors.append(pos_similarity * 0.25)
        
        # 2. Similitud de trauma type (25%)
        type_similarity = 1.0 if scar1.trauma_type == scar2.trauma_type else 0.0
        factors.append(type_similarity * 0.25)
        
        # 3. Similitud de lliçons (30%)
        if scar1.lessons_learned and scar2.lessons_learned:
            set1 = set(scar1.lessons_learned)
            set2 = set(scar2.lessons_learned)
            
            if set1 or set2:
                jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
                factors.append(jaccard * 0.3)
            else:
                factors.append(0.0)
        else:
            factors.append(0.0)
        
        # 4. Similitud temporal (20%)
        time_diff = abs((scar1.timestamp - scar2.timestamp).total_seconds())
        time_similarity = 1.0 / (1.0 + time_diff / 3600)  # Decau en hores
        factors.append(time_similarity * 0.2)
        
        return sum(factors)
    
    def _consolidate_similar_scars(self, scar_group: List[str]) -> int:
        """Consolida un grup de cicatrius similars."""
        if len(scar_group) < 2:
            return 0
        
        scars = [self.scar_archive.scars[sid] for sid in scar_group 
                if sid in self.scar_archive.scars]
        
        if len(scars) < 2:
            return 0
        
        # Crear cicatriu consolidada
        avg_position = np.mean([s.position for s in scars], axis=0)
        avg_potential = np.mean([s.evolutionary_potential for s in scars])
        
        # Combinar lliçons úniques
        all_lessons = []
        for scar in scars:
            all_lessons.extend(scar.lessons_learned)
        unique_lessons = list(set(all_lessons))
        
        # Crear nova cicatriu
        from core.svgelona_engine_v5_2 import EvolutionaryScar
        import hashlib
        
        consolidated_id = hashlib.sha256(
            f"consolidated_{'_'.join(sorted(scar_group))}".encode()
        ).hexdigest()[:16]
        
        consolidated_scar = EvolutionaryScar(
            scar_id=consolidated_id,
            trauma_type=f"consolidated_{scars[0].trauma_type}",
            position=avg_position,
            timestamp=datetime.now(),
            lessons_learned=unique_lessons,
            evolutionary_potential=avg_potential,
            access_count=sum(s.access_count for s in scars),
            last_accessed=max(s.last_accessed for s in scars)
        )
        
        # Eliminar cicatrius originals
        memory_saved = 0
        for scar_id in scar_group:
            if scar_id in self.scar_archive.scars:
                # Estimar memòria alliberada
                scar = self.scar_archive.scars[scar_id]
                memory_saved += self._estimate_memory_footprint(scar)
                
                # Eliminar
                self.scar_archive.remove_scar(scar_id)
                if scar_id in self.memory_items:
                    del self.memory_items[scar_id]
        
        # Afegir cicatriu consolidada
        self.scar_archive.add_scar(consolidated_scar)
        self._add_scar_to_memory(consolidated_scar)
        
        return memory_saved
    
    def _reprioritize_items(self):
        """Reajusta prioritats basant-se en ús recent."""
        for item_id, item in self.memory_items.items():
            if item_id not in self.scar_archive.scars:
                continue
            
            scar = self.scar_archive.scars[item_id]
            
            # Recalcular prioritat i utilitat
            new_priority = self._calculate_scar_priority(scar)
            new_utility = self._calculate_scar_utility(scar)
            
            # Actualitzar item
            item.priority = new_priority
            item.utility_score = new_utility
            item.last_accessed = scar.last_accessed
            item.access_count = scar.access_count
            
            # Recalcular clau d'ordenació
            item.__post_init__()
        
        # Reconstruir heap
        self.priority_queue = []
        for item_id, item in self.memory_items.items():
            heapq.heappush(self.priority_queue, (item.sort_key, item_id))
    
    def _cleanup_priority_queue(self):
        """Neteja el heap de prioritats d'items eliminats."""
        # Crear nou heap només amb items existents
        new_queue = []
        
        for sort_key, item_id in self.priority_queue:
            if item_id in self.memory_items:
                heapq.heappush(new_queue, (sort_key, item_id))
        
        self.priority_queue = new_queue
    
    def get_memory_manager_report(self) -> Dict[str, Any]:
        """Genera informe complet del gestor de memòria."""
        scar_stats = self.scar_archive.get_archive_stats()
        
        # Distribució per prioritat
        priority_dist = {p.name: 0 for p in MemoryPriority}
        for item in self.memory_items.values():
            priority_dist[item.priority.name] += 1
        
        # Mètriques d'utilitat
        utility_scores = [item.utility_score for item in self.memory_items.values()]
        
        return {
            "memory_management": {
                **self.stats,
                "config": {
                    "max_memory_mb": self.config["max_memory_mb"],
                    "target_utilization": self.config["target_memory_utilization"],
                    "min_utility_threshold": self.config["min_utility_threshold"]
                },
                "current_pressure": self.check_memory_pressure()
            },
            "priority_distribution": priority_dist,
            "utility_metrics": {
                "avg_utility": np.mean(utility_scores) if utility_scores else 0,
                "min_utility": min(utility_scores) if utility_scores else 0,
                "max_utility": max(utility_scores) if utility_scores else 0,
                "low_utility_items": sum(1 for u in utility_scores if u < 0.3)
            },
            "scar_archive": scar_stats
        }
    
    def mark_as_critical(self, item_id: str):
        """Marca un item com a crític (no es pot eliminar)."""
        self.config["critical_items"].add(item_id)
        
        # Actualitzar prioritat si existeix
        if item_id in self.memory_items:
            self.memory_items[item_id].priority = MemoryPriority.CRITICAL
    
    def optimize_for_performance(self, target_items: int = 1000) -> Dict[str, Any]:
        """
        Optimitza el sistema per a un nombre objectiu d'items.
        
        Redueix memòria i millora rendiment ajustant paràmetres.
        """
        optimization_report = {
            "target_items": target_items,
            "current_items": self.stats["total_items"],
            "actions": []
        }
        
        # Si tenim massa items, aplicar evicció agressiva
        if self.stats["total_items"] > target_items * 1.5:
            items_to_remove = self.stats["total_items"] - target_items
            
            # Ajustar llindar d'utilitat
            old_threshold = self.config["min_utility_threshold"]
            self.config["min_utility_threshold"] = 0.4  # Més agressiu
            
            optimization_report["actions"].append(
                f"adjusted_utility_threshold: {old_threshold} -> 0.4"
            )
            
            # Executar gestió de memòria
            mgmt_report = self.perform_memory_management()
            optimization_report["management_result"] = mgmt_report
            
            # Restaurar llindar
            self.config["min_utility_threshold"] = old_threshold
        
        # Optimitzar cerca
        if "performance_metrics" in self.scar_archive.stats:
            cache_hit_rate = self.scar_archive.stats["performance_metrics"]["cache_hit_rate"]
            
            if cache_hit_rate < 0.5:
                # Incrementar cache size
                old_limit = self.scar_archive.cache_size_limit
                self.scar_archive.cache_size_limit = 2000
                
                optimization_report["actions"].append(
                    f"increased_cache_size: {old_limit} -> 2000"
                )
        
        optimization_report["final_items"] = self.stats["total_items"]
        optimization_report["final_memory_mb"] = self.stats["memory_used_mb"]
        
        return optimization_report