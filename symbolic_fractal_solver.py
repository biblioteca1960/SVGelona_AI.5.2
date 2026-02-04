"""
SVGelona_AI 5.2 - Solucionador Simbòlic Fractal
Optimització de fractals mitjançant anàlisi simbòlica.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import sympy as sp
from dataclasses import dataclass, field
from datetime import datetime
import math

@dataclass
class FractalEquation:
    """Equació simbòlica que descriu un fractal."""
    
    equation_id: str
    variables: List[sp.Symbol]
    equations: List[sp.Eq]
    complexity: float
    fitness_score: float = 0.0
    parameters: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class OptimizationResult:
    """Resultat d'una optimització fractal."""
    
    success: bool
    optimal_parameters: Dict[str, float]
    fitness_before: float
    fitness_after: float
    iterations: int
    convergence_rate: float
    constraints_violated: List[str] = field(default_factory=list)

class SymbolicFractalSolver:
    """
    Solucionador que utilitza anàlisi simbòlica per optimitzar fractals.
    """
    
    def __init__(self):
        # Variables simbòliques
        self.x, self.y, self.z = sp.symbols('x y z', real=True)
        self.r = sp.Symbol('r', positive=True)  # Radi
        self.theta = sp.Symbol('θ', real=True)  # Angle
        self.phi = sp.Symbol('φ', real=True)    # Angle azimutal
        self.d = sp.Symbol('d', positive=True)  # Distància
        
        # Llibreria d'equacions fractals
        self.fractal_library: Dict[str, FractalEquation] = {}
        self._initialize_fractal_library()
        
        # Configuració
        self.config = {
            "max_iterations": 100,
            "convergence_threshold": 1e-6,
            "learning_rate": 0.1,
            "constraint_weight": 10.0,
            "symbolic_simplification": True,
            "numeric_precision": 1e-8
        }
        
        # Cache de derivades
        self.derivative_cache: Dict[Tuple, sp.Expr] = {}
        
        # Estadístiques
        self.stats = {
            "optimizations_performed": 0,
            "successful_optimizations": 0,
            "avg_iterations": 0,
            "avg_fitness_improvement": 0.0
        }
    
    def _initialize_fractal_library(self):
        """Inicialitza la llibreria amb equacions fractals conegudes."""
        
        # 1. Sistema de Lindenmayer simple (L-system)
        lsystem_eq = FractalEquation(
            equation_id="lsystem_basic",
            variables=[self.x, self.y, self.theta, self.d],
            equations=[
                sp.Eq(self.x, self.x + self.d * sp.cos(self.theta)),
                sp.Eq(self.y, self.y + self.d * sp.sin(self.theta)),
                sp.Eq(self.theta, self.theta + sp.pi/4)  # Rotació de 45°
            ],
            complexity=1.0,
            parameters={"d": 1.0, "theta_0": 0.0}
        )
        self.fractal_library["lsystem_basic"] = lsystem_eq
        
        # 2. Attractor de Lorenz (simplificat)
        lorenz_eq = FractalEquation(
            equation_id="lorenz_simplified",
            variables=[self.x, self.y, self.z],
            equations=[
                sp.Eq(sp.diff(self.x, self.d), 10*(self.y - self.x)),
                sp.Eq(sp.diff(self.y, self.d), self.x*(28 - self.z) - self.y),
                sp.Eq(sp.diff(self.z, self.d), self.x*self.y - (8/3)*self.z)
            ],
            complexity=2.5,
            parameters={}
        )
        self.fractal_library["lorenz_simplified"] = lorenz_eq
        
        # 3. Conjunt de Mandelbrot (formulació simbòlica)
        mandelbrot_eq = FractalEquation(
            equation_id="mandelbrot_symbolic",
            variables=[self.x, self.y],
            equations=[
                sp.Eq(self.x, self.x**2 - self.y**2 + self.r),
                sp.Eq(self.y, 2*self.x*self.y + self.phi)
            ],
            complexity=2.0,
            parameters={"r": 0.0, "phi": 0.0}
        )
        self.fractal_library["mandelbrot_symbolic"] = mandelbrot_eq
        
        # 4. Arbre fractal (optimització geomètrica)
        fractal_tree_eq = FractalEquation(
            equation_id="fractal_tree",
            variables=[self.x, self.y, self.theta, self.d],
            equations=[
                # Branca principal
                sp.Eq(self.x, self.x + self.d * sp.cos(self.theta)),
                sp.Eq(self.y, self.y + self.d * sp.sin(self.theta)),
                # Branques secundàries (recursiu)
                sp.Eq(self.theta, self.theta + sp.pi/6),  # +30°
                sp.Eq(sp.Symbol('theta2'), self.theta - sp.pi/6)  # -30°
            ],
            complexity=1.8,
            parameters={"d": 1.0, "branching_factor": 0.7}
        )
        self.fractal_library["fractal_tree"] = fractal_tree_eq
    
    def generate_optimal_fractal(self,
                               seed_position: np.ndarray,
                               depth: int = 6,
                               base_params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Genera un fractal optimitzat a partir d'una posició inicial.
        
        Args:
            seed_position: Posició inicial (llavor)
            depth: Profunditat del fractal
            base_params: Paràmetres base per a l'optimització
            
        Returns:
            Resultat de l'optimització
        """
        if base_params is None:
            base_params = {"r": 1.0, "θ": 0.0, "d": 1.0, "φ": 0.0}
        
        # Seleccionar equació fractal basant-se en la llavor
        equation = self._select_fractal_equation(seed_position)
        
        # Configurar paràmetres inicials
        initial_params = base_params.copy()
        initial_params.update({
            "x0": float(seed_position[0]) if len(seed_position) > 0 else 0.0,
            "y0": float(seed_position[1]) if len(seed_position) > 1 else 0.0,
            "z0": float(seed_position[2]) if len(seed_position) > 2 else 0.0
        })
        
        # Avaluar fitness inicial
        fitness_before = self._evaluate_fractal_fitness(
            equation, initial_params, depth
        )
        
        # Optimitzar paràmetres
        optimization_result = self._optimize_parameters(
            equation, initial_params, depth
        )
        
        # Avaluar fitness després de l'optimització
        if optimization_result.success:
            fitness_after = self._evaluate_fractal_fitness(
                equation, optimization_result.optimal_parameters, depth
            )
            
            # Actualitzar estadístiques
            self.stats["optimizations_performed"] += 1
            self.stats["successful_optimizations"] += 1
            self.stats["avg_iterations"] = (
                (self.stats["avg_iterations"] * (self.stats["successful_optimizations"] - 1) +
                 optimization_result.iterations) / self.stats["successful_optimizations"]
            )
            
            improvement = (fitness_after - fitness_before) / (abs(fitness_before) + 1e-10)
            self.stats["avg_fitness_improvement"] = (
                (self.stats["avg_fitness_improvement"] * (self.stats["successful_optimizations"] - 1) +
                 improvement) / self.stats["successful_optimizations"]
            )
            
            # Generar fractal amb paràmetres optimitzats
            optimal_fractal = self._generate_fractal_from_equation(
                equation, optimization_result.optimal_parameters, depth
            )
            
            return {
                "optimization_success": True,
                "equation_used": equation.equation_id,
                "fitness_improvement": fitness_after - fitness_before,
                "fitness_before": fitness_before,
                "fitness_after": fitness_after,
                "optimal_parameters": optimization_result.optimal_parameters,
                "iterations": optimization_result.iterations,
                "fractal_data": optimal_fractal,
                "collisions_before": self._count_collisions(equation, initial_params, depth),
                "collisions_after": self._count_collisions(equation, optimization_result.optimal_parameters, depth),
                "optimal_angles_applied": self._extract_optimal_angles(optimization_result.optimal_parameters)
            }
        else:
            return {
                "optimization_success": False,
                "equation_used": equation.equation_id,
                "error": "L'optimització no va convergir",
                "constraints_violated": optimization_result.constraints_violated,
                "fractal_data": self._generate_fractal_from_equation(equation, initial_params, depth)
            }
    
    def _select_fractal_equation(self, seed_position: np.ndarray) -> FractalEquation:
        """Selecciona l'equació fractal més adequada per a una llavor."""
        
        # Basar-se en característiques de la llavor
        seed_norm = np.linalg.norm(seed_position)
        seed_entropy = self._calculate_position_entropy(seed_position)
        
        # Heurístiques de selecció
        if seed_norm < 0.5:
            # Prop de l'origen → Mandelbrot
            return self.fractal_library["mandelbrot_symbolic"]
        elif seed_entropy > 0.7:
            # Alta entropia → Lorenz
            return self.fractal_library["lorenz_simplified"]
        elif len(seed_position) >= 3 and abs(seed_position[2]) > 0.1:
            # Component Z significatiu → Arbre 3D
            return self.fractal_library["fractal_tree"]
        else:
            # Per defecte → L-system
            return self.fractal_library["lsystem_basic"]
    
    def _calculate_position_entropy(self, position: np.ndarray) -> float:
        """Calcula l'entropia d'una posició."""
        if len(position) < 2:
            return 0.0
        
        # Discretitzar components
        hist, _ = np.histogram(position, bins=min(10, len(position)))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        
        if len(hist) < 2:
            return 0.0
        
        # Entropia de Shannon
        entropy = -np.sum(hist * np.log2(hist))
        max_entropy = np.log2(len(hist))
        
        return float(entropy / max_entropy)
    
    def _evaluate_fractal_fitness(self,
                                equation: FractalEquation,
                                parameters: Dict[str, float],
                                depth: int) -> float:
        """
        Avalua el fitness d'un fractal (més alt és millor).
        
        Factors:
        1. Complexitat controlada
        2. Minimització de col·lisions
        3. Estabilitat numèrica
        4. Bellesa (heurística)
        """
        fitness_components = []
        
        # 1. Complexitat adequada (ni massa simple ni massa complexa)
        target_complexity = depth * 0.5
        complexity_score = 1.0 / (1.0 + abs(equation.complexity - target_complexity))
        fitness_components.append(complexity_score * 0.3)
        
        # 2. Minimització de col·lisions
        collision_count = self._count_collisions(equation, parameters, depth)
        collision_score = np.exp(-collision_count / 10.0)  # Decau exponencialment
        fitness_components.append(collision_score * 0.3)
        
        # 3. Estabilitat numèrica
        stability = self._evaluate_numerical_stability(equation, parameters, depth)
        fitness_components.append(stability * 0.2)
        
        # 4. Simetria (heurística de bellesa)
        symmetry = self._evaluate_symmetry(equation, parameters, depth)
        fitness_components.append(symmetry * 0.2)
        
        return sum(fitness_components)
    
    def _count_collisions(self,
                         equation: FractalEquation,
                         parameters: Dict[str, float],
                         depth: int,
                         threshold: float = 0.1) -> int:
        """Compta col·lisions entre branques fractals."""
        
        # Generar punts del fractal
        points = self._generate_fractal_points(equation, parameters, depth)
        
        if len(points) < 2:
            return 0
        
        # Comptar parelles massa properes
        collision_count = 0
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distance = np.linalg.norm(points[i] - points[j])
                if distance < threshold:
                    collision_count += 1
        
        return collision_count
    
    def _generate_fractal_points(self,
                               equation: FractalEquation,
                               parameters: Dict[str, float],
                               depth: int) -> np.ndarray:
        """Genera punts d'un fractal a partir d'una equació."""
        
        points = []
        
        # Extraure punts inicials
        x0 = parameters.get("x0", 0.0)
        y0 = parameters.get("y0", 0.0)
        z0 = parameters.get("z0", 0.0)
        
        current_point = np.array([x0, y0, z0])
        points.append(current_point)
        
        # Iterar l'equació (simplificat)
        for i in range(depth * 10):  # Generar múltiples punts per profunditat
            # Aplicar transformacions basades en l'equació
            if equation.equation_id == "lsystem_basic":
                # Simular L-system
                d = parameters.get("d", 1.0)
                theta = parameters.get("θ", 0.0) + i * np.pi/4
                
                new_point = current_point + np.array([
                    d * np.cos(theta),
                    d * np.sin(theta),
                    0
                ])
                
            elif equation.equation_id == "mandelbrot_symbolic":
                # Simular iteració de Mandelbrot
                r = parameters.get("r", 0.0)
                phi = parameters.get("φ", 0.0)
                
                x, y, z = current_point
                new_x = x**2 - y**2 + r
                new_y = 2*x*y + phi
                new_z = z * 0.9  # Petita contracció en Z
                
                new_point = np.array([new_x, new_y, new_z])
            
            else:
                # Transformació genèrica
                noise = np.random.normal(0, 0.1, 3)
                new_point = current_point * 0.9 + noise
            
            points.append(new_point)
            current_point = new_point
        
        return np.array(points)
    
    def _evaluate_numerical_stability(self,
                                    equation: FractalEquation,
                                    parameters: Dict[str, float],
                                    depth: int) -> float:
        """Avalua l'estabilitat numèrica del fractal."""
        
        points = self._generate_fractal_points(equation, parameters, depth)
        
        if len(points) < 2:
            return 0.5
        
        # Verificar divergència
        norms = np.linalg.norm(points, axis=1)
        max_norm = np.max(norms)
        
        if max_norm > 1000:  # Divergència ràpida
            return 0.1
        
        # Verificar oscil·lació excessiva
        diffs = np.diff(norms)
        oscillation = np.std(diffs) / (np.mean(np.abs(diffs)) + 1e-10)
        
        if oscillation > 5.0:
            return 0.3
        
        # Estabilitat bona
        return 0.9
    
    def _evaluate_symmetry(self,
                          equation: FractalEquation,
                          parameters: Dict[str, float],
                          depth: int) -> float:
        """Avalua la simetria del fractal (heurística de bellesa)."""
        
        points = self._generate_fractal_points(equation, parameters, depth)
        
        if len(points) < 10:
            return 0.5
        
        # Verificar simetria respecte als eixos
        symmetries = []
        
        # Simetria X
        x_symmetry = np.mean(np.abs(points[:, 0] + points[:, 0])) / (np.mean(np.abs(points[:, 0])) + 1e-10)
        symmetries.append(min(1.0, x_symmetry))
        
        # Simetria Y
        y_symmetry = np.mean(np.abs(points[:, 1] + points[:, 1])) / (np.mean(np.abs(points[:, 1])) + 1e-10)
        symmetries.append(min(1.0, y_symmetry))
        
        # Simetria rotacional (aproximada)
        angles = np.arctan2(points[:, 1], points[:, 0])
        angle_hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
        angle_hist = angle_hist / angle_hist.sum()
        rotational_symmetry = 1.0 - np.std(angle_hist) / np.mean(angle_hist)
        symmetries.append(max(0.0, rotational_symmetry))
        
        return float(np.mean(symmetries))
    
    def _optimize_parameters(self,
                           equation: FractalEquation,
                           initial_params: Dict[str, float],
                           depth: int) -> OptimizationResult:
        """
        Optimitza els paràmetres d'una equació fractal.
        
        Utilitza gradient descent simbòlic.
        """
        
        # Identificar paràmetres optimitzables (excloent posicions inicials)
        optimizable_params = [k for k in initial_params.keys() 
                            if k not in ["x0", "y0", "z0"]]
        
        if not optimizable_params:
            return OptimizationResult(
                success=False,
                optimal_parameters=initial_params,
                fitness_before=0.0,
                fitness_after=0.0,
                iterations=0,
                convergence_rate=0.0,
                constraints_violated=["No hi ha paràmetres optimitzables"]
            )
        
        # Inicialitzar
        params = initial_params.copy()
        learning_rate = self.config["learning_rate"]
        
        best_params = params.copy()
        best_fitness = self._evaluate_fractal_fitness(equation, params, depth)
        
        iteration = 0
        converged = False
        constraints_violated = []
        
        while iteration < self.config["max_iterations"] and not converged:
            iteration += 1
            
            # Calcular gradient per a cada paràmetre
            gradients = {}
            
            for param_name in optimizable_params:
                # Derivada numèrica
                epsilon = 1e-5
                params_plus = params.copy()
                params_plus[param_name] += epsilon
                
                params_minus = params.copy()
                params_minus[param_name] -= epsilon
                
                fitness_plus = self._evaluate_fractal_fitness(equation, params_plus, depth)
                fitness_minus = self._evaluate_fractal_fitness(equation, params_minus, depth)
                
                gradient = (fitness_plus - fitness_minus) / (2 * epsilon)
                gradients[param_name] = gradient
            
            # Actualitzar paràmetres
            old_params = params.copy()
            
            for param_name, gradient in gradients.items():
                params[param_name] += learning_rate * gradient
            
            # Verificar restriccions
            constraints_violated = self._check_constraints(equation, params)
            
            if constraints_violated:
                # Violació de restriccions → retrocedir
                params = old_params
                learning_rate *= 0.5  # Reduir taxa d'aprenentatge
                continue
            
            # Avaluar nou fitness
            current_fitness = self._evaluate_fractal_fitness(equation, params, depth)
            
            # Actualitzar millors paràmetres
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_params = params.copy()
            
            # Verificar convergència
            if iteration > 1:
                improvement = current_fitness - best_fitness
                if abs(improvement) < self.config["convergence_threshold"]:
                    converged = True
            
            # Ajustar taxa d'aprenentatge
            if iteration % 10 == 0:
                learning_rate *= 0.95  # Decaïment gradual
        
        # Calcular taxa de convergència
        convergence_rate = best_fitness / (iteration + 1e-10)
        
        return OptimizationResult(
            success=converged,
            optimal_parameters=best_params,
            fitness_before=self._evaluate_fractal_fitness(equation, initial_params, depth),
            fitness_after=best_fitness,
            iterations=iteration,
            convergence_rate=convergence_rate,
            constraints_violated=constraints_violated
        )
    
    def _check_constraints(self,
                          equation: FractalEquation,
                          parameters: Dict[str, float]) -> List[str]:
        """Verifica restriccions en els paràmetres."""
        
        constraints_violated = []
        
        # Restriccions basades en l'equació
        if equation.equation_id == "mandelbrot_symbolic":
            # r ha de ser petit per a estabilitat
            r = parameters.get("r", 0.0)
            if abs(r) > 2.0:
                constraints_violated.append(f"r={r} massa gran per a Mandelbrot")
        
        # Restriccions generals
        for param_name, value in parameters.items():
            if param_name == "d" and value <= 0:
                constraints_violated.append(f"d={value} ha de ser positiu")
            elif param_name == "θ" and abs(value) > 2*np.pi:
                constraints_violated.append(f"θ={value} fora del rang [-2π, 2π]")
        
        return constraints_violated
    
    def _generate_fractal_from_equation(self,
                                      equation: FractalEquation,
                                      parameters: Dict[str, float],
                                      depth: int) -> Dict[str, Any]:
        """Genera dades estructurades d'un fractal."""
        
        points = self._generate_fractal_points(equation, parameters, depth)
        
        # Calcular métriques
        bounds = {
            "min": points.min(axis=0).tolist(),
            "max": points.max(axis=0).tolist(),
            "center": points.mean(axis=0).tolist()
        }
        
        # Calcular fractal dimension (aproximada)
        fractal_dimension = self._estimate_fractal_dimension(points)
        
        return {
            "equation_id": equation.equation_id,
            "point_count": len(points),
            "bounds": bounds,
            "fractal_dimension": fractal_dimension,
            "sample_points": points[:100].tolist(),  # Limitar per a l'informe
            "parameters_used": parameters
        }
    
    def _estimate_fractal_dimension(self, points: np.ndarray) -> float:
        """Estima la dimensió fractal mitjançant el mètode box-counting."""
        
        if len(points) < 100:
            return 1.0  # Valor per defecte per a pocs punts
        
        # Implementació simplificada de box-counting
        scales = [0.1, 0.05, 0.025, 0.0125]
        counts = []
        
        for scale in scales:
            # Discretitzar espai
            min_vals = points.min(axis=0)
            max_vals = points.max(axis=0)
            
            # Calcular nombre de caixes
            num_boxes = np.prod(np.ceil((max_vals - min_vals) / scale).astype(int))
            counts.append(num_boxes)
        
        # Ajustar recta log-log
        if len(scales) >= 2:
            log_scales = np.log(1/np.array(scales))
            log_counts = np.log(counts)
            
            # Regressió lineal
            coeffs = np.polyfit(log_scales, log_counts, 1)
            dimension = coeffs[0]  # Pendent = dimensió fractal
        else:
            dimension = 1.0
        
        return float(min(3.0, max(1.0, dimension)))
    
    def _extract_optimal_angles(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Extrau angles òptims dels paràmetres."""
        
        optimal_angles = {}
        
        for key, value in parameters.items():
            if key in ["θ", "phi", "φ", "angle", "rotation"]:
                # Convertir a graus
                optimal_angles[key] = math.degrees(value) if isinstance(value, (int, float)) else 0.0
        
        return optimal_angles
    
    def get_solver_report(self) -> Dict[str, Any]:
        """Genera informe del solucionador simbòlic."""
        
        return {
            "solver_statistics": self.stats,
            "equation_library": {
                eq_id: {
                    "complexity": eq.complexity,
                    "variables": [str(v) for v in eq.variables],
                    "fitness_score": eq.fitness_score
                }
                for eq_id, eq in self.fractal_library.items()
            },
            "configuration": self.config,
            "cache_performance": {
                "derivative_cache_size": len(self.derivative_cache)
            }
        }