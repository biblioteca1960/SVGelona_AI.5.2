"""
SVGelona_AI 5.2 - Sistema Principal Optimitzat
Sistema d'IA generativa fractal amb gestió de memòria integrada.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.svgelona_engine_v5_2 import (
    SVGelonaEngine, 
    OptimizedScarArchive, 
    OptimizedFractalModule
)
from core.integrated_memory_manager import IntegratedMemoryManager
from core.axioms_bridge_theorems import AxiomBridgeEngine
from core.angular_geometry import AngularGeometryEngine
from optimization.symbolic_fractal_solver import SymbolicFractalSolver
from optimization.css_matrix_transformer import CSSMatrixTransformer

import json
from datetime import datetime
import numpy as np
import time
from typing import Dict, List, Any, Optional

class SVGelonaAI5_2:
    """
    SVGelona_AI 5.2 - Sistema complet optimitzat.
    
    Característiques principals:
    1. Gestió de memòria integrada amb poda selectiva
    2. Indexació espacial per a cerca ràpida
    3. Generació fractal optimitzada fins a profunditat 12
    4. Solucionador simbòlic per a optimització de fractals
    5. Transformacions CSS optimitzades per GPU
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicialitza el sistema SVGelona_AI 5.2.
        
        Args:
            config: Configuració personalitzada
        """
        print("=" * 70)
        print("INICIALITZANT SVGELONA_AI 5.2")
        print("Sistema d'IA generativa fractal optimitzat")
        print("=" * 70)
        
        # Configuració per defecte
        self.config = {
            "max_scars": 10000,           # Límit de cicatrius
            "max_fractal_depth": 12,      # Profunditat fractal màxima
            "memory_limit_mb": 100,       # Límit de memòria
            "performance_mode": "balanced", # balanced, quality, performance
            "auto_optimize": True,        # Optimització automàtica
            "save_state_interval": 100,   # Guardar estat cada N generacions
            "render_enabled": True        # Habilitar renderització
        }
        
        # Sobreescriure amb configuració personalitzada
        if config:
            self.config.update(config)
        
        # Inicialitzar components optimitzats
        print("\n[1/6] Inicialitzant components del sistema...")
        
        # Motor principal amb arxiu optimitzat
        self.scar_archive = OptimizedScarArchive(cell_size=2.0)
        self.engine = SVGelonaEngine()
        self.engine.scar_archive = self.scar_archive  # Reemplaçar amb arxiu optimitzat
        
        # Mòduls optimitzats
        self.fractal_module = OptimizedFractalModule()
        self.fractal_module.max_depth = self.config["max_fractal_depth"]
        
        # Sistemes d'optimització
        print("[2/6] Inicialitzant sistemes d'optimització...")
        self.axiom_bridge = AxiomBridgeEngine(self.scar_archive)
        self.angular_geometry = AngularGeometryEngine()
        self.memory_manager = IntegratedMemoryManager(
            self.scar_archive, 
            self.axiom_bridge
        )
        self.symbolic_solver = SymbolicFractalSolver()
        self.css_transformer = CSSMatrixTransformer()
        
        # Configurar gestor de memòria
        self.memory_manager.config["max_memory_mb"] = self.config["memory_limit_mb"]
        
        # Historial del sistema
        self.generation_count = 0
        self.system_history: List[Dict[str, Any]] = []
        self.performance_log: List[Dict[str, Any]] = []
        
        # Estadístiques en temps real
        self.realtime_stats = {
            "generations_per_second": 0.0,
            "avg_processing_time": 0.0,
            "memory_usage_mb": 0.0,
            "scar_growth_rate": 0.0
        }
        
        # Timers
        self.last_generation_time = time.time()
        self.start_time = time.time()
        
        print("[3/6] Configurant mode de rendiment...")
        self._configure_performance_mode()
        
        print("[4/6] Carregant estat inicial...")
        self._load_initial_state()
        
        print("[5/6] Realitzant comprovacions del sistema...")
        self._run_system_checks()
        
        print("[6/6] Sistema inicialitzat correctament!")
        print("=" * 70)
        
        self._print_system_summary()
    
    def _configure_performance_mode(self):
        """Configura el sistema segons el mode de rendiment seleccionat."""
        mode = self.config["performance_mode"]
        
        if mode == "performance":
            # Maximitzar rendiment
            self.fractal_module.optimization_params.update({
                "branch_pruning_threshold": 0.05,
                "adaptive_depth": True,
                "batch_processing": True
            })
            
            self.memory_manager.config.update({
                "min_utility_threshold": 0.3,
                "eviction_batch_size": 20
            })
            
            print("  Mode: RENDIMENT (màxima velocitat)")
            
        elif mode == "quality":
            # Maximitzar qualitat
            self.fractal_module.optimization_params.update({
                "branch_pruning_threshold": 0.0,
                "adaptive_depth": False,
                "batch_processing": False
            })
            
            self.memory_manager.config.update({
                "min_utility_threshold": 0.1,
                "eviction_batch_size": 5
            })
            
            print("  Mode: QUALITAT (màxima precisió)")
            
        else:  # balanced
            # Equilibri entre rendiment i qualitat
            self.fractal_module.optimization_params.update({
                "branch_pruning_threshold": 0.01,
                "adaptive_depth": True,
                "batch_processing": True
            })
            
            print("  Mode: EQUILIBRAT (balanç optimitzat)")
    
    def _load_initial_state(self):
        """Carrega estat inicial o crea un de nou."""
        # Intentar carregar estat guardat
        try:
            if os.path.exists("svgelona_state.json"):
                with open("svgelona_state.json", 'r') as f:
                    state = json.load(f)
                
                # Carregar cicatrius (simplificat)
                print(f"  Estat carregat: {len(state.get('scars', []))} cicatrius")
            else:
                print("  No s'ha trobat estat guardat, començant de zero")
                
        except Exception as e:
            print(f"  Error carregant estat: {e}, començant de zero")
    
    def _run_system_checks(self):
        """Executa comprovacions del sistema."""
        checks_passed = 0
        total_checks = 4
        
        # Check 1: Components inicialitzats
        if all([
            self.engine is not None,
            self.scar_archive is not None,
            self.memory_manager is not None
        ]):
            checks_passed += 1
            print("  ✓ Components del sistema: OK")
        else:
            print("  ✗ Components del sistema: ERROR")
        
        # Check 2: Memòria disponible
        import psutil
        memory = psutil.virtual_memory()
        available_mb = memory.available / (1024 * 1024)
        
        if available_mb > self.config["memory_limit_mb"] * 2:
            checks_passed += 1
            print(f"  ✓ Memòria disponible: {available_mb:.0f} MB")
        else:
            print(f"  ⚠ Memòria limitada: {available_mb:.0f} MB")
        
        # Check 3: Rendiment inicial
        try:
            # Prova de rendiment ràpida
            start = time.time()
            test_result = self.fractal_module.spawn_fractal_manifold(
                np.array([0.0, 0.0, 0.0]),
                depth=5
            )
            elapsed = time.time() - start
            
            if elapsed < 1.0:
                checks_passed += 1
                print(f"  ✓ Rendiment inicial: {elapsed:.3f}s per fractal")
            else:
                print(f"  ⚠ Rendiment lent: {elapsed:.3f}s per fractal")
                
        except Exception as e:
            print(f"  ✗ Error prova de rendiment: {e}")
        
        # Check 4: Espai d'emmagatzematge
        try:
            import shutil
            total, used, free = shutil.disk_usage(".")
            free_gb = free / (1024**3)
            
            if free_gb > 1.0:
                checks_passed += 1
                print(f"  ✓ Espai d'emmagatzematge: {free_gb:.1f} GB lliures")
            else:
                print(f"  ⚠ Espai d'emmagatzematge limitat: {free_gb:.1f} GB")
                
        except:
            print("  ⚠ No s'ha pogut verificar l'espai d'emmagatzematge")
        
        print(f"  {checks_passed}/{total_checks} comprovacions superades")
    
    def _print_system_summary(self):
        """Imprimeix resum del sistema."""
        print("\n" + "=" * 70)
        print("RESUM DEL SISTEMA SVGELONA_AI 5.2")
        print("=" * 70)
        
        scar_count = len(self.scar_archive.scars)
        axiom_count = len(self.axiom_bridge.axioms)
        
        print(f"Components actius:")
        print(f"  • Arxiu de cicatrius: {scar_count} cicatrius")
        print(f"  • Sistema axiomàtic: {axiom_count} axiomes")
        print(f"  • Gestor de memòria: {self.config['memory_limit_mb']} MB límit")
        print(f"  • Profunditat fractal: fins a {self.config['max_fractal_depth']} nivells")
        
        print(f"\nConfiguració de rendiment:")
        print(f"  • Mode: {self.config['performance_mode']}")
        print(f"  • Optimització automàtica: {'SI' if self.config['auto_optimize'] else 'NO'}")
        print(f"  • Renderització: {'ACTIVA' if self.config['render_enabled'] else 'INACTIVA'}")
        
        # Informació de rendiment
        if self.performance_log:
            latest = self.performance_log[-1]
            print(f"\nRendiment actual:")
            print(f"  • Generacions/segon: {self.realtime_stats['generations_per_second']:.2f}")
            print(f"  • Ús de memòria: {self.realtime_stats['memory_usage_mb']:.1f} MB")
        
        print("=" * 70)
    
    def run_generation(self, 
                      steps: int = 5,
                      optimize: bool = True) -> Dict[str, Any]:
        """
        Executa una generació completa del sistema.
        
        Args:
            steps: Nombre de passos d'evolució per generació
            optimize: Aplicar optimitzacions després de la generació
            
        Returns:
            Diccionari amb resultats de la generació
        """
        self.generation_count += 1
        gen_start_time = time.time()
        
        print(f"\n[GENERACIÓ {self.generation_count}]")
        print(f"Hora: {datetime.now().strftime('%H:%M:%S')}")
        print(f"Passos: {steps}")
        print("-" * 50)
        
        # PAS 1: Evolucionar sistema principal
        print("PAS 1: Evolució del sistema...")
        evolution_results = self._run_optimized_evolution(steps)
        
        # PAS 2: Gestió de memòria
        print("PAS 2: Gestió de memòria...")
        memory_report = self.memory_manager.perform_memory_management()
        
        # PAS 3: Processar traumes i derivar axiomes
        print("PAS 3: Processament de traumes...")
        trauma_report = self._process_traumas()
        
        # PAS 4: Actualitzar geometria angular
        print("PAS 4: Actualització angular...")
        angular_report = self._update_angular_geometry()
        
        # PAS 5: Optimització si està activada
        optimization_report = {}
        if optimize and self.config["auto_optimize"]:
            print("PAS 5: Optimització del sistema...")
            optimization_report = self._run_optimizations()
        
        # PAS 6: Generar fractal optimitzat
        print("PAS 6: Generació fractal...")
        fractal_report = self._generate_optimized_fractal()
        
        # Calcular mètriques de rendiment
        gen_time = time.time() - gen_start_time
        self._update_performance_metrics(gen_time, steps)
        
        # Compilar resultats
        generation_result = {
            "generation": self.generation_count,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": gen_time,
            "performance_metrics": self.realtime_stats.copy(),
            "evolution": evolution_results,
            "memory": memory_report,
            "trauma_processing": trauma_report,
            "angular_geometry": angular_report,
            "optimization": optimization_report,
            "fractal_generation": fractal_report,
            "system_state": self._get_system_state_summary()
        }
        
        # Guardar en historial
        self.system_history.append(generation_result)
        
        # Guardar estat periòdicament
        if self.generation_count % self.config["save_state_interval"] == 0:
            self.save_system_state()
        
        # Mostrar resum
        self._print_generation_summary(generation_result)
        
        return generation_result
    
    def _run_optimized_evolution(self, steps: int) -> Dict[str, Any]:
        """
        Executa evolució optimitzada del sistema.
        
        Utilitza totes les optimitzacions implementades.
        """
        results = []
        
        for step in range(steps):
            step_start = time.time()
            
            try:
                # Evolucionar estat
                self.engine._apply_evolutionary_pressure()
                self.engine._update_state()
                self.engine._check_bounds()
                
                # Generar fractal amb mòdul optimitzat
                manifold = self.fractal_module.spawn_fractal_manifold(
                    self.engine.state.position,
                    depth=min(8, 3 + int(self.engine.state.coherence * 5))
                )
                
                step_result = {
                    "step": step,
                    "position": self.engine.state.position.tolist(),
                    "energy": self.engine.state.energy,
                    "coherence": self.engine.state.coherence,
                    "entropy": self.engine.state.entropy,
                    "fractal_complexity": len(manifold["branches"]),
                    "duration": time.time() - step_start
                }
                
                results.append(step_result)
                
            except Exception as e:
                # Registrar error com a trauma
                self.engine._record_trauma("optimized_evolution_error", str(e))
                print(f"  ⚠ Error en pas {step}: {e}")
        
        return {
            "steps_completed": len(results),
            "average_duration_per_step": np.mean([r["duration"] for r in results]) if results else 0,
            "final_state": {
                "position": self.engine.state.position.tolist(),
                "coherence": self.engine.state.coherence,
                "entropy": self.engine.state.entropy
            }
        }
    
    def _process_traumas(self) -> Dict[str, Any]:
        """Processa traumes i deriva nous axiomes."""
        # Simular alguns traumes per a demostració
        simulated_traumas = [
            {
                "type": "boundary_collision",
                "data": {
                    "position": self.engine.state.position.tolist(),
                    "distance": np.linalg.norm(self.engine.state.position)
                }
            }
        ]
        
        axioms_derived = []
        for trauma in simulated_traumas:
            new_axioms = self.axiom_bridge.process_trauma(
                trauma["type"],
                trauma["data"]
            )
            axioms_derived.extend(new_axioms)
        
        return {
            "traumas_processed": len(simulated_traumas),
            "axioms_derived": len(axioms_derived),
            "total_axioms": len(self.axiom_bridge.axioms)
        }
    
    def _update_angular_geometry(self) -> Dict[str, Any]:
        """Actualitza la geometria angular del sistema."""
        angular_state = self.angular_geometry.update_angular_state(
            self.engine.state.position,
            self.engine.state.momentum
        )
        
        return {
            "angular_state": {
                "phase": angular_state.phase.name,
                "angular_entropy": angular_state.angular_entropy,
                "torsion_magnitude": float(np.linalg.norm(angular_state.torsion_tensor))
            },
            "svg_transform": self.angular_geometry.generate_angular_svg_transform()
        }
    
    def _run_optimizations(self) -> Dict[str, Any]:
        """Executa optimitzacions del sistema."""
        optimizations = []
        
        # Optimització de memòria si hi ha pressió
        memory_pressure = self.memory_manager.check_memory_pressure()
        
        if memory_pressure > 0.8:
            opt_result = self.memory_manager.optimize_for_performance(
                target_items=min(1000, len(self.scar_archive.scars) * 2 // 3)
            )
            optimizations.append(("memory_optimization", opt_result))
        
        # Optimització de fractals periòdica
        if self.generation_count % 10 == 0:
            # Utilitzar solucionador simbòlic per optimitzar un fractal
            seed = self.engine.state.position
            fractal_result = self.symbolic_solver.generate_optimal_fractal(
                seed_position=seed,
                depth=6,
                base_params={"r": 1.2, "θ": 0.5, "d": 1.0}
            )
            
            if fractal_result.get("optimization_success", False):
                optimizations.append(("fractal_optimization", {
                    "collisions_reduced": fractal_result["collisions_before"] - fractal_result["collisions_after"],
                    "optimal_angles_applied": fractal_result["optimal_angles_applied"]
                }))
        
        return {
            "optimizations_applied": len(optimizations),
            "details": optimizations,
            "memory_pressure": memory_pressure
        }
    
    def _generate_optimized_fractal(self) -> Dict[str, Any]:
        """Genera un fractal optimitzat."""
        # Generar fractal amb profunditat adaptativa
        coherence = self.engine.state.coherence
        adaptive_depth = max(3, min(self.config["max_fractal_depth"], 
                                   int(3 + coherence * 6)))
        
        manifold = self.fractal_module.spawn_fractal_manifold(
            self.engine.state.position,
            depth=adaptive_depth,
            adaptive=True
        )
        
        # Aplicar transformacions CSS si la renderització està activada
        css_transform = None
        if self.config["render_enabled"]:
            torsion_tensor = self.angular_geometry.state.torsion_tensor
            css_transform = self.css_transformer.torsion_tensor_to_css_matrix(
                torsion_tensor,
                self.engine.state.position,
                phase=self.angular_geometry.state.phase.name.lower()
            )
        
        return {
            "depth": adaptive_depth,
            "branch_count": manifold["branch_count"],
            "entropy": manifold["entropy"],
            "cache_performance": manifold["cache_info"],
            "css_transform": css_transform.to_css_string() if css_transform else None
        }
    
    def _update_performance_metrics(self, gen_time: float, steps: int):
        """Actualitza mètriques de rendiment en temps real."""
        now = time.time()
        
        # Generacions per segon
        if gen_time > 0:
            gens_per_sec = steps / gen_time
        else:
            gens_per_sec = 0
        
        # Mitjana mòbil
        self.realtime_stats["generations_per_second"] = (
            self.realtime_stats["generations_per_second"] * 0.7 + gens_per_sec * 0.3
        )
        
        # Temps de processament mitjà
        self.realtime_stats["avg_processing_time"] = (
            self.realtime_stats["avg_processing_time"] * 0.7 + gen_time * 0.3
        )
        
        # Ús de memòria
        mem_report = self.memory_manager.get_memory_manager_report()
        self.realtime_stats["memory_usage_mb"] = mem_report["memory_management"]["memory_used_mb"]
        
        # Taxa de creixement de cicatrius
        scar_count = len(self.scar_archive.scars)
        if hasattr(self, 'last_scar_count'):
            growth = scar_count - self.last_scar_count
            self.realtime_stats["scar_growth_rate"] = growth
        self.last_scar_count = scar_count
        
        # Guardar en log de rendiment
        self.performance_log.append({
            "timestamp": datetime.now().isoformat(),
            "generation": self.generation_count,
            "duration": gen_time,
            "gens_per_sec": gens_per_sec,
            "memory_mb": self.realtime_stats["memory_usage_mb"],
            "scar_count": scar_count
        })
        
        # Mantenir log manejable
        if len(self.performance_log) > 1000:
            self.performance_log = self.performance_log[-1000:]
    
    def _get_system_state_summary(self) -> Dict[str, Any]:
        """Genera resum de l'estat del sistema."""
        scar_stats = self.scar_archive.get_archive_stats()
        mem_report = self.memory_manager.get_memory_manager_report()
        fractal_stats = self.fractal_module.get_module_stats()
        angular_report = self.angular_geometry.get_angular_report()
        
        return {
            "generation_count": self.generation_count,
            "uptime_hours": (time.time() - self.start_time) / 3600,
            "scar_archive": {
                "total_scars": scar_stats.get("total_scars", 0),
                "performance": scar_stats.get("performance_metrics", {})
            },
            "memory_management": mem_report.get("memory_management", {}),
            "fractal_module": fractal_stats,
            "angular_geometry": angular_report.get("angular_state", {}),
            "axiom_system": {
                "total_axioms": len(self.axiom_bridge.axioms),
                "consistency": self.axiom_bridge.check_axiom_consistency().get("consistency_score", 0)
            }
        }
    
    def _print_generation_summary(self, generation_result: Dict[str, Any]):
        """Imprimeix resum d'una generació."""
        print(f"\n{'='*50}")
        print(f"RESUM GENERACIÓ {generation_result['generation']}")
        print(f"{'='*50}")
        
        # Informació bàsica
        duration = generation_result["duration_seconds"]
        print(f"Durada: {duration:.3f}s")
        print(f"Rendiment: {generation_result['performance_metrics']['generations_per_second']:.2f} gens/s")
        
        # Evolució
        evolution = generation_result["evolution"]
        print(f"Evolució: {evolution['steps_completed']} passos")
        print(f"Coherència final: {evolution['final_state']['coherence']:.3f}")
        print(f"Entropia final: {evolution['final_state']['entropy']:.3f}")
        
        # Memòria
        memory = generation_result["memory"]
        if "items_affected" in memory:
            evicted = memory["items_affected"].get("evicted_count", 0)
            if evicted > 0:
                print(f"Memòria: {evicted} cicatrius eliminades")
        
        # Fractal
        fractal = generation_result["fractal_generation"]
        print(f"Fractal: profunditat {fractal['depth']}, {fractal['branch_count']} branques")
        
        # Ús de memòria
        memory_mb = generation_result["performance_metrics"]["memory_usage_mb"]
        print(f"Ús memòria: {memory_mb:.1f} MB")
        
        print(f"{'='*50}")
    
    def save_system_state(self, filename: str = "svgelona_state_v5_2.json"):
        """Guarda l'estat actual del sistema."""
        state = {
            "version": "SVGelona_AI_5.2",
            "timestamp": datetime.now().isoformat(),
            "generation_count": self.generation_count,
            "engine_state": {
                "position": self.engine.state.position.tolist(),
                "momentum": self.engine.state.momentum.tolist(),
                "energy": self.engine.state.energy,
                "coherence": self.engine.state.coherence,
                "entropy": self.engine.state.entropy
            },
            "scar_archive": {
                "scar_count": len(self.scar_archive.scars),
                "stats": self.scar_archive.get_archive_stats()
            },
            "memory_manager": self.memory_manager.get_memory_manager_report(),
            "system_config": self.config
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            print(f"\n💾 Estat del sistema guardat a '{filename}'")
            return True
            
        except Exception as e:
            print(f"\n⚠ Error guardant estat: {e}")
            return False
    
    def run_benchmark(self, generations: int = 10, steps_per_gen: int = 5) -> Dict[str, Any]:
        """
        Executa un benchmark del sistema.
        
        Args:
            generations: Nombre de generacions per executar
            steps_per_gen: Passos per generació
            
        Returns:
            Resultats del benchmark
        """
        print(f"\n{'='*70}")
        print(f"INICIANT BENCHMARK")
        print(f"Generacions: {generations}")
        print(f"Passos per generació: {steps_per_gen}")
        print(f"{'='*70}")
        
        benchmark_start = time.time()
        results = []
        
        for gen in range(1, generations + 1):
            print(f"\nBenchmark generació {gen}/{generations}...")
            
            gen_start = time.time()
            result = self.run_generation(
                steps=steps_per_gen,
                optimize=False  # Desactivar optimització per benchmark pur
            )
            gen_time = time.time() - gen_start
            
            results.append({
                "generation": gen,
                "duration": gen_time,
                "scar_count": len(self.scar_archive.scars),
                "memory_mb": result["performance_metrics"]["memory_usage_mb"]
            })
        
        total_time = time.time() - benchmark_start
        
        # Calcular estadístiques
        durations = [r["duration"] for r in results]
        scar_counts = [r["scar_count"] for r in results]
        memory_usage = [r["memory_mb"] for r in results]
        
        benchmark_result = {
            "benchmark_config": {
                "generations": generations,
                "steps_per_gen": steps_per_gen,
                "performance_mode": self.config["performance_mode"]
            },
            "performance": {
                "total_time": total_time,
                "avg_time_per_gen": np.mean(durations),
                "std_time_per_gen": np.std(durations),
                "total_generations_per_second": generations / total_time,
                "avg_scar_growth_per_gen": np.mean(np.diff(scar_counts)) if len(scar_counts) > 1 else 0,
                "final_memory_usage_mb": memory_usage[-1] if memory_usage else 0,
                "memory_growth_mb": (memory_usage[-1] - memory_usage[0]) if len(memory_usage) > 1 else 0
            },
            "system_state_after": self._get_system_state_summary()
        }
        
        print(f"\n{'='*70}")
        print(f"BENCHMARK COMPLETAT")
        print(f"{'='*70}")
        print(f"Temps total: {total_time:.2f}s")
        print(f"Temps mitjà per generació: {benchmark_result['performance']['avg_time_per_gen']:.3f}s")
        print(f"Generacions per segon: {benchmark_result['performance']['total_generations_per_second']:.2f}")
        print(f"Cicatrius finals: {scar_counts[-1] if scar_counts else 0}")
        print(f"Ús de memòria final: {benchmark_result['performance']['final_memory_usage_mb']:.1f} MB")
        print(f"{'='*70}")
        
        return benchmark_result
    
    def generate_visualization_report(self) -> Dict[str, Any]:
        """
        Genera un informe de visualització amb dades per a gràfics.
        """
        if not self.system_history:
            return {"error": "No hi ha dades d'historial"}
        
        # Agafar dades recents
        recent_history = self.system_history[-100:]  # Últimes 100 generacions
        
        # Extraure sèries temporals
        generations = [h["generation"] for h in recent_history]
        coherences = [h["evolution"]["final_state"]["coherence"] for h in recent_history]
        entropies = [h["evolution"]["final_state"]["entropy"] for h in recent_history]
        memory_usage = [h["performance_metrics"]["memory_usage_mb"] for h in recent_history]
        fractal_complexities = [h["fractal_generation"]["branch_count"] for h in recent_history]
        
        # Calcular tendències
        from scipy import stats
        
        trends = {}
        
        if len(generations) > 1:
            # Tendència de coherència
            coherence_slope, _, _, _, _ = stats.linregress(generations, coherences)
            trends["coherence"] = {
                "slope": coherence_slope,
                "direction": "increasing" if coherence_slope > 0 else "decreasing",
                "strength": abs(coherence_slope)
            }
            
            # Tendència d'entropia
            entropy_slope, _, _, _, _ = stats.linregress(generations, entropies)
            trends["entropy"] = {
                "slope": entropy_slope,
                "direction": "increasing" if entropy_slope > 0 else "decreasing",
                "strength": abs(entropy_slope)
            }
        
        # Agrupar per fase angular
        angular_phases = {}
        for h in recent_history:
            phase = h["angular_geometry"]["angular_state"]["phase"]
            angular_phases[phase] = angular_phases.get(phase, 0) + 1
        
        return {
            "time_series": {
                "generations": generations,
                "coherences": coherences,
                "entropies": entropies,
                "memory_usage": memory_usage,
                "fractal_complexities": fractal_complexities
            },
            "trends": trends,
            "phase_distribution": angular_phases,
            "current_state": self._get_system_state_summary(),
            "performance_metrics": self.realtime_stats
        }

def main():
    """Funció principal d'execució."""
    print("SVGelona_AI 5.2 - Sistema Principal")
    print("Versió optimitzada amb gestió de memòria integrada")
    print()
    
    # Configuració
    config = {
        "max_scars": 5000,
        "max_fractal_depth": 10,
        "memory_limit_mb": 50,
        "performance_mode": "balanced",
        "auto_optimize": True,
        "save_state_interval": 50,
        "render_enabled": True
    }
    
    # Crear instància del sistema
    system = SVGelonaAI5_2(config)
    
    # Executar algunes generacions
    print("\nExecutant 5 generacions inicials...")
    for _ in range(5):
        system.run_generation(steps=3)
    
    # Executar benchmark
    print("\nExecutant benchmark de rendiment...")
    benchmark = system.run_benchmark(generations=5, steps_per_gen=3)
    
    # Generar informe de visualització
    print("\nGenerant informe de visualització...")
    viz_report = system.generate_visualization_report()
    
    # Guardar resultats
    with open("benchmark_results.json", "w") as f:
        json.dump(benchmark, f, indent=2, default=str)
    
    with open("visualization_data.json", "w") as f:
        json.dump(viz_report, f, indent=2, default=str)
    
    # Guardar estat final
    system.save_system_state()
    
    print("\n" + "="*70)
    print("EXECUCIÓ COMPLETADA")
    print("="*70)
    print(f"Generacions executades: {system.generation_count}")
    print(f"Cicatrius acumulades: {len(system.scar_archive.scars)}")
    print(f"Ús de memòria final: {system.realtime_stats['memory_usage_mb']:.1f} MB")
    print(f"Rendiment final: {system.realtime_stats['generations_per_second']:.2f} gens/s")
    print("="*70)
    print("Resultats guardats a:")
    print("  • benchmark_results.json")
    print("  • visualization_data.json")
    print("  • svgelona_state_v5_2.json")
    print("="*70)

if __name__ == "__main__":
    main()