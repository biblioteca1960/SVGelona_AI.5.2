"""
SVGelona_AI 5.2 - Motor Principal Optimitzat
Sistema d'IA generativa fractal amb gestió de memòria i rendiment.
"""
import numpy as np
import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
import math
from collections import defaultdict

@dataclass
class FractalState:
    """Estat actual del sistema fractal (optimitzat)."""
    position: np.ndarray  # Posició en espai fractal
    momentum: np.ndarray  # Momentum evolutiu
    energy: float         # Energia fractal disponible
    coherence: float      # Coherència euclidiana (0-1)
    entropy: float        # Entropia de torsió (0-1)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if self.position is None:
            self.position = np.zeros(3, dtype=np.float32)
        if self.momentum is None:
            self.momentum = np.ones(3, dtype=np.float32) * 0.1
        if self.energy <= 0:
            self.energy = 1.0

@dataclass
class EvolutionaryScar:
    """Cicatriu d'un trauma evolutiu superat (optimitzada)."""
    scar_id: str
    trauma_type: str           # Tipus de trauma
    position: np.ndarray       # On va passar (float32 per estalvi)
    timestamp: datetime
    lessons_learned: List[str] # Lliçons extretes (comprimides)
    evolutionary_potential: float  # Potencial d'evolució (0-1)
    access_count: int = 0      # Vegades accedida
    last_accessed: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if self.lessons_learned is None:
            self.lessons_learned = []
        if self.evolutionary_potential <= 0:
            self.evolutionary_potential = 0.5
        if self.position is None:
            self.position = np.zeros(3, dtype=np.float32)

class OptimizedScarArchive:
    """
    Arxiu de cicatrius amb indexació espacial per a cerca ràpida.
    
    Utilitza un grid espacial per a cerques O(1) en lloc de O(N).
    """
    
    def __init__(self, cell_size: float = 2.0):
        self.scars: Dict[str, EvolutionaryScar] = {}
        self.learning_rate = 0.1
        
        # Grid espacial per a cerca ràpida
        self.cell_size = cell_size
        self.spatial_grid: Dict[Tuple[int, int, int], Set[str]] = defaultdict(set)
        
        # Cache de cerques freqüents
        self.search_cache: Dict[Tuple, List[str]] = {}
        self.cache_size_limit = 1000
        
        # Estadístiques
        self.stats = {
            "total_scars": 0,
            "searches_performed": 0,
            "cache_hits": 0,
            "grid_hits": 0
        }
    
    def _position_to_cell(self, position: np.ndarray) -> Tuple[int, int, int]:
        """Converteix posició a coordenades de cel·la del grid."""
        cell_x = int(position[0] / self.cell_size)
        cell_y = int(position[1] / self.cell_size)
        cell_z = int(position[2] / self.cell_size)
        return (cell_x, cell_y, cell_z)
    
    def _get_nearby_cells(self, center_cell: Tuple[int, int, int], 
                         radius: int = 1) -> List[Tuple[int, int, int]]:
        """Obté cel·les properes a una cel·la donada."""
        cells = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    cells.append((center_cell[0] + dx,
                                 center_cell[1] + dy,
                                 center_cell[2] + dz))
        return cells
    
    def add_scar(self, scar: EvolutionaryScar):
        """Afegeix una nova cicatriu al sistema indexat."""
        self.scars[scar.scar_id] = scar
        
        # Afegir al grid espacial
        cell = self._position_to_cell(scar.position)
        self.spatial_grid[cell].add(scar.scar_id)
        
        self.stats["total_scars"] += 1
        
        # Invalidar cache si cal
        if len(self.search_cache) > self.cache_size_limit:
            # Eliminar entrades més antigues
            keys_to_remove = list(self.search_cache.keys())[:self.cache_size_limit // 2]
            for key in keys_to_remove:
                del self.search_cache[key]
    
    def find_relevant_scars(self, 
                           position: np.ndarray, 
                           radius: float = 1.0) -> List[EvolutionaryScar]:
        """
        Troba cicatrius properes a una posició.
        
        Complexitat: O(1) amb cache, O(k) amb grid, on k << N
        """
        self.stats["searches_performed"] += 1
        
        # Crear clau de cache
        cache_key = (tuple(np.round(position, 2)), radius)
        
        # Verificar cache
        if cache_key in self.search_cache:
            self.stats["cache_hits"] += 1
            scar_ids = self.search_cache[cache_key]
            return [self.scars[sid] for sid in scar_ids if sid in self.scars]
        
        # Cerca utilitzant grid espacial
        center_cell = self._position_to_cell(position)
        
        # Calcular radi en cel·les
        cell_radius = max(1, int(np.ceil(radius / self.cell_size)))
        
        # Buscar en cel·les properes
        nearby_cells = self._get_nearby_cells(center_cell, cell_radius)
        
        scar_ids = set()
        for cell in nearby_cells:
            if cell in self.spatial_grid:
                self.stats["grid_hits"] += 1
                scar_ids.update(self.spatial_grid[cell])
        
        # Filtrar per distància real (dins del radi)
        relevant_scars = []
        for scar_id in scar_ids:
            if scar_id in self.scars:
                scar = self.scars[scar_id]
                distance = np.linalg.norm(scar.position - position)
                if distance <= radius:
                    relevant_scars.append(scar)
                    scar.access_count += 1
                    scar.last_accessed = datetime.now()
        
        # Actualitzar cache
        scar_id_list = [s.scar_id for s in relevant_scars]
        self.search_cache[cache_key] = scar_id_list
        
        return relevant_scars
    
    def get_evolutionary_pressure(self, position: np.ndarray) -> float:
        """Calcula pressió evolutiva basada en cicatrius properes."""
        scars = self.find_relevant_scars(position)
        if not scars:
            return 0.0
        
        # Pes per access_count i potencial
        total_weight = sum(
            scar.evolutionary_potential * (1 + np.log1p(scar.access_count))
            for scar in scars
        )
        
        avg_pressure = total_weight / len(scars)
        
        # Normalitzar entre 0 i 1
        return min(1.0, avg_pressure)
    
    def remove_scar(self, scar_id: str) -> bool:
        """Elimina una cicatriu del sistema."""
        if scar_id not in self.scars:
            return False
        
        scar = self.scars[scar_id]
        
        # Eliminar del grid
        cell = self._position_to_cell(scar.position)
        if cell in self.spatial_grid:
            self.spatial_grid[cell].discard(scar_id)
            if not self.spatial_grid[cell]:
                del self.spatial_grid[cell]
        
        # Eliminar del diccionari principal
        del self.scars[scar_id]
        
        # Eliminar del cache (costós, però necessari)
        keys_to_remove = []
        for key, scar_ids in self.search_cache.items():
            if scar_id in scar_ids:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.search_cache[key]
        
        self.stats["total_scars"] = max(0, self.stats["total_scars"] - 1)
        
        return True
    
    def get_low_utility_scars(self, 
                             min_access_count: int = 1,
                             max_age_days: int = 7) -> List[str]:
        """
        Identifica cicatrius de baixa utilitat per a possible poda.
        
        Criteris:
        1. Poques vegades accedida
        2. Molt antiga
        3. Baix potencial evolutiu
        """
        now = datetime.now()
        low_utility = []
        
        for scar_id, scar in self.scars.items():
            # Calcular puntuació d'utilitat
            age_days = (now - scar.timestamp).total_seconds() / (24 * 3600)
            
            utility_score = (
                scar.access_count * 0.4 +
                (1.0 - min(1.0, age_days / max_age_days)) * 0.3 +
                scar.evolutionary_potential * 0.3
            )
            
            if utility_score < 0.3:  # Llindar baix
                low_utility.append(scar_id)
        
        return low_utility
    
    def get_archive_stats(self) -> Dict[str, Any]:
        """Retorna estadístiques de l'arxiu."""
        if not self.scars:
            return {"total_scars": 0, "empty": True}
        
        access_counts = [s.access_count for s in self.scars.values()]
        potentials = [s.evolutionary_potential for s in self.scars.values()]
        
        now = datetime.now()
        ages = []
        for scar in self.scars.values():
            age_hours = (now - scar.timestamp).total_seconds() / 3600
            ages.append(age_hours)
        
        return {
            **self.stats,
            "performance_metrics": {
                "cache_hit_rate": self.stats["cache_hits"] / max(1, self.stats["searches_performed"]),
                "avg_access_count": np.mean(access_counts) if access_counts else 0,
                "avg_potential": np.mean(potentials) if potentials else 0,
                "avg_age_hours": np.mean(ages) if ages else 0,
                "grid_cell_count": len(self.spatial_grid),
                "cache_size": len(self.search_cache)
            },
            "scar_distribution": {
                "by_trauma_type": defaultdict(int, {
                    scar.trauma_type: sum(1 for s in self.scars.values() 
                                         if s.trauma_type == scar.trauma_type)
                    for scar in self.scars.values()
                })
            }
        }

class OptimizedFractalModule:
    """Mòdul de generació fractal optimitzat per a profunditats altes."""
    
    def __init__(self):
        self.max_depth = 12  # Incrementat de 5 a 12
        self.growth_rate = 1.15  # Reduït per a estabilitat
        self.complexity_limit = 5000
        
        # Cache de patrons fractals
        self.pattern_cache: Dict[Tuple, List[Dict]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Paràmetres optimitzats
        self.optimization_params = {
            "branch_pruning_threshold": 0.01,  # Branques molt petites es poden
            "angle_optimization": True,        # Optimitzar angles per a eficiència
            "adaptive_depth": True,            # Ajustar profunditat segons complexitat
            "batch_processing": True           # Processar en lots per eficiència
        }
    
    def spawn_fractal_manifold(self, 
                              seed: np.ndarray,
                              depth: int = 5,
                              adaptive: bool = True) -> Dict[str, Any]:
        """
        Genera una varietat fractal optimitzada.
        
        Utilitza cache i tècniques d'optimització per a profunditats altes.
        """
        # Ajustar profunditat si és adaptativa
        if adaptive and self.optimization_params["adaptive_depth"]:
            depth = self._calculate_adaptive_depth(seed, depth)
        
        # Verificar cache
        cache_key = self._create_cache_key(seed, depth)
        if cache_key in self.pattern_cache:
            self.cache_hits += 1
            cached_branches = self.pattern_cache[cache_key]
        else:
            self.cache_misses += 1
            cached_branches = self._generate_optimized_branches(seed, depth)
            self.pattern_cache[cache_key] = cached_branches
            
            # Netejar cache si és massa gran
            if len(self.pattern_cache) > 1000:
                self._cleanup_cache()
        
        # Calcular mètriques
        entropy = self._calculate_optimized_entropy(seed, cached_branches)
        
        manifold = {
            "seed": seed.tolist(),
            "depth": depth,
            "branch_count": len(cached_branches),
            "branches": cached_branches,
            "entropy": entropy,
            "euclidean_distance": float(np.linalg.norm(seed)),
            "cache_info": {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses)
            }
        }
        
        return manifold
    
    def _create_cache_key(self, seed: np.ndarray, depth: int) -> Tuple:
        """Crea una clau de cache per a un fractal."""
        # Arrodonir per a agrupar patrons similars
        seed_rounded = tuple(np.round(seed, 2))
        return (seed_rounded, depth, self.growth_rate)
    
    def _calculate_adaptive_depth(self, seed: np.ndarray, requested_depth: int) -> int:
        """Calcula profunditat òptima basant-se en la complexitat."""
        base_depth = min(requested_depth, self.max_depth)
        
        # Reduir profunditat si el fractal serà massa complex
        estimated_complexity = 2 ** base_depth
        
        if estimated_complexity > self.complexity_limit:
            # Trobar profunditat màxima que no excedeixi el límit
            max_allowed_depth = int(np.log2(self.complexity_limit))
            return min(base_depth, max_allowed_depth)
        
        # Ajustar basant-se en l'entropia de la llavor
        seed_entropy = self._calculate_seed_entropy(seed)
        
        # Més entropia → menor profunditat per a estabilitat
        entropy_factor = 1.0 - seed_entropy * 0.3
        
        adaptive_depth = int(base_depth * entropy_factor)
        
        return max(2, min(adaptive_depth, self.max_depth))
    
    def _calculate_seed_entropy(self, seed: np.ndarray) -> float:
        """Calcula entropia de la llavor (mesura d'aleatorietat)."""
        flat = seed.flatten()
        if len(flat) < 2:
            return 0.0
        
        # Histograma normalitzat
        hist, _ = np.histogram(flat, bins=min(10, len(flat)))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        
        if len(hist) < 2:
            return 0.0
        
        # Entropia de Shannon
        entropy = -np.sum(hist * np.log2(hist))
        max_entropy = np.log2(len(hist))
        
        return float(entropy / max_entropy)
    
    def _generate_optimized_branches(self, 
                                   seed: np.ndarray, 
                                   depth: int) -> List[Dict]:
        """
        Genera branques fractals optimitzades.
        
        Tècniques:
        1. Branch pruning (eliminar branques insignificant)
        2. Angle optimization (minimizar col·lisions)
        3. Batch processing (vectorització)
        """
        branches = []
        
        # Vectorització per a eficiència
        branch_count = 2 ** depth
        
        if self.optimization_params["batch_processing"] and branch_count > 16:
            # Processament per lots per a grans fractals
            batch_size = min(64, branch_count)
            
            for batch_start in range(0, branch_count, batch_size):
                batch_end = min(batch_start + batch_size, branch_count)
                
                batch_branches = self._generate_branch_batch(
                    seed, depth, batch_start, batch_end
                )
                
                branches.extend(batch_branches)
        else:
            # Processament seqüencial per a fractals petits
            rng = np.random.default_rng(int(abs(seed[0]) * 1000))
            
            for i in range(branch_count):
                branch = self._generate_single_branch(seed, depth, i, rng)
                
                # Aplicar branch pruning
                if self._should_keep_branch(branch):
                    branches.append(branch)
        
        return branches
    
    def _generate_branch_batch(self, 
                             seed: np.ndarray,
                             depth: int,
                             start_idx: int,
                             end_idx: int) -> List[Dict]:
        """Genera un lot de branques de manera vectoritzada."""
        batch_size = end_idx - start_idx
        
        # Generar vectors aleatoris de manera vectoritzada
        rng = np.random.default_rng(int(abs(seed[0]) * 1000) + start_idx)
        random_vectors = rng.normal(0, 1, (batch_size, 3))
        
        # Normalitzar
        norms = np.linalg.norm(random_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Evitar divisió per zero
        unit_vectors = random_vectors / norms
        
        # Aplicar creixement fractal
        growth_factors = self.growth_rate ** depth
        scaled_vectors = unit_vectors * growth_factors
        
        branches = []
        for i in range(batch_size):
            branch_vector = scaled_vectors[i]
            
            # Optimitzar angle si està activat
            if self.optimization_params["angle_optimization"]:
                branch_vector = self._optimize_branch_angle(
                    branch_vector, seed, depth, start_idx + i
                )
            
            length = float(np.linalg.norm(branch_vector))
            angle = math.atan2(branch_vector[1], branch_vector[0])
            
            branches.append({
                "vector": branch_vector.tolist(),
                "length": length,
                "angle": angle,
                "depth_level": depth,
                "branch_index": start_idx + i
            })
        
        return branches
    
    def _generate_single_branch(self, 
                              seed: np.ndarray,
                              depth: int,
                              index: int,
                              rng: np.random.Generator) -> Dict[str, Any]:
        """Genera una única branca."""
        branch_vector = rng.normal(0, 1, 3)
        branch_vector = branch_vector / np.linalg.norm(branch_vector)
        
        # Aplicar creixement fractal
        length = self.growth_rate ** depth
        branch_vector *= length
        
        # Optimitzar angle si està activat
        if self.optimization_params["angle_optimization"]:
            branch_vector = self._optimize_branch_angle(branch_vector, seed, depth, index)
        
        angle = math.atan2(branch_vector[1], branch_vector[0])
        
        return {
            "vector": branch_vector.tolist(),
            "length": length,
            "angle": angle,
            "depth_level": depth,
            "branch_index": index
        }
    
    def _optimize_branch_angle(self, 
                              branch_vector: np.ndarray,
                              seed: np.ndarray,
                              depth: int,
                              index: int) -> np.ndarray:
        """Optimitza l'angle d'una branca per evitar col·lisions."""
        # Angle base
        base_angle = math.atan2(branch_vector[1], branch_vector[0])
        
        # Ajustar basant-se en la profunditat i índex
        # Fractals més profunds tenen angles més distribuïts
        angle_adjustment = (index / (2 ** depth)) * 2 * math.pi
        
        # Aplicar ajust
        new_angle = base_angle + angle_adjustment * 0.1
        
        # Reconstruir vector amb nou angle
        length = np.linalg.norm(branch_vector)
        new_vector = np.array([
            length * math.cos(new_angle),
            length * math.sin(new_angle),
            branch_vector[2]  # Mantenir component Z
        ])
        
        return new_vector
    
    def _should_keep_branch(self, branch: Dict[str, Any]) -> bool:
        """Determina si s'ha de conservar una branca (branch pruning)."""
        if not self.optimization_params["branch_pruning_threshold"]:
            return True
        
        length = branch["length"]
        threshold = self.optimization_params["branch_pruning_threshold"]
        
        # Eliminar branques massa petites
        if length < threshold:
            return False
        
        # Branques amb angles estranys poden indicar problemes numèrics
        angle = branch["angle"]
        if abs(angle) > math.pi * 0.9:  # Gairebé 180 graus
            return False
        
        return True
    
    def _calculate_optimized_entropy(self, 
                                   seed: np.ndarray,
                                   branches: List[Dict]) -> float:
        """Calcula entropia optimitzada per a un conjunt de branques."""
        if not branches:
            return 0.0
        
        # Utilitzar múltiples mètriques per a entropia
        metrics = []
        
        # 1. Variabilitat de longituds
        lengths = [b["length"] for b in branches]
        if lengths:
            length_cv = np.std(lengths) / (np.mean(lengths) + 1e-10)
            metrics.append(min(1.0, length_cv))
        
        # 2. Variabilitat d'angles
        angles = [b["angle"] for b in branches]
        if angles:
            # Normalitzar angles
            angles_norm = [(a + math.pi) % (2 * math.pi) for a in angles]
            angle_std = np.std(angles_norm)
            metrics.append(min(1.0, angle_std / math.pi))
        
        # 3. Entropia de Shannon de les direccions
        if len(branches) >= 4:
            # Discretitzar direccions
            direction_bins = 8
            direction_counts = [0] * direction_bins
            
            for branch in branches:
                vector = np.array(branch["vector"])
                if np.linalg.norm(vector) > 0:
                    direction = vector / np.linalg.norm(vector)
                    
                    # Convertir a coordenades esfèriques
                    phi = math.atan2(direction[1], direction[0])  # Azimut
                    bin_idx = int((phi + math.pi) / (2 * math.pi) * direction_bins) % direction_bins
                    direction_counts[bin_idx] += 1
            
            # Calcular entropia
            probs = np.array(direction_counts) / sum(direction_counts)
            probs = probs[probs > 0]
            
            if len(probs) > 1:
                shannon_entropy = -np.sum(probs * np.log2(probs))
                max_entropy = np.log2(len(probs))
                metrics.append(shannon_entropy / max_entropy)
        
        return float(np.mean(metrics)) if metrics else 0.5
    
    def _cleanup_cache(self):
        """Neteja la cache, eliminant entrades poc utilitzades."""
        if not self.pattern_cache:
            return
        
        # Per ara, eliminem aleatòriament la meitat
        # En una implementació real, usaríem LRU o similar
        keys = list(self.pattern_cache.keys())
        keys_to_remove = keys[:len(keys) // 2]
        
        for key in keys_to_remove:
            del self.pattern_cache[key]
    
    def get_module_stats(self) -> Dict[str, Any]:
        """Retorna estadístiques del mòdul fractal."""
        return {
            "max_depth": self.max_depth,
            "growth_rate": self.growth_rate,
            "complexity_limit": self.complexity_limit,
            "cache_performance": {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses),
                "cache_size": len(self.pattern_cache)
            },
            "optimization_params": self.optimization_params
        }