"""
SVGelona_AI 5.2 - Motor de Geometria Angular (AMB ESTABILITZACIÓ SVD)
Gestió de fases angulars, torsió i transformacions SVG amb protecció contra singularitats.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
import math

class AngularPhase(Enum):
    """Fases del cicle angular."""
    EMERGENCE = "emergence"      # Emergència d'estructures
    COHERENCE = "coherence"      # Estabilització coherent
    TORSION = "torsion"          # Torsió i deformació
    INTEGRATION = "integration"  # Integració de patrons
    TRANSITION = "transition"    # Transició entre fases

@dataclass
class AngularState:
    """Estat actual de la geometria angular."""
    
    phase: AngularPhase
    angular_momentum: np.ndarray  # Momentum angular (3D)
    torsion_tensor: np.ndarray    # Tensor de torsió (3x3)
    phase_progress: float         # Progrés dins de la fase (0-1)
    angular_entropy: float        # Entropia angular (0-1)
    structural_stability: float   # Estabilitat estructural (0-1, 1=perfecta)
    last_phase_transition: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if self.angular_momentum is None:
            self.angular_momentum = np.zeros(3, dtype=np.float32)
        if self.torsion_tensor is None:
            self.torsion_tensor = np.eye(3, dtype=np.float32)
        if self.phase_progress < 0 or self.phase_progress > 1:
            self.phase_progress = 0.0
        if self.angular_entropy < 0 or self.angular_entropy > 1:
            self.angular_entropy = 0.5
        if self.structural_stability < 0 or self.structural_stability > 1:
            self.structural_stability = 1.0

class AngularGeometryEngine:
    """
    Motor de geometria angular amb estabilització SVD.
    Implementa normalització simplèctica per evitar singularitats.
    """
    
    def __init__(self):
        self.state = AngularState(
            phase=AngularPhase.EMERGENCE,
            angular_momentum=np.zeros(3),
            torsion_tensor=np.eye(3),
            phase_progress=0.0,
            angular_entropy=0.5,
            structural_stability=1.0
        )
        
        # Configuració
        self.config = {
            "phase_duration_range": (50, 200),     # Rang de durada de fase en cicles
            "torsion_strength": 0.1,               # Força màxima de torsió
            "angular_damping": 0.95,                # Amortiment de momentum angular
            "entropy_threshold": 0.8,              # Llindar per a transició de fase
            "max_angular_velocity": 2.0,           # Velocitat angular màxima
            
            # Paràmetres d'estabilització SVD
            "svd_correction_enabled": True,        # Habilitar correcció SVD
            "min_singular_value": 1e-6,            # Valor singular mínim
            "preserve_volume": True,               # Mantenir determinant = 1
            "stability_check_interval": 10,        # Cicles entre verificacions
            "adaptive_correction": True,           # Correcció adaptativa
            "structural_decay_rate": 0.99,         # Taxa de decaïment de l'estabilitat
            "stability_recovery_rate": 0.05        # Taxa de recuperació de l'estabilitat
        }
        
        # Historial de fases
        self.phase_history: List[Dict[str, Any]] = []
        
        # Paràmetres de fase
        self.current_phase_duration = 100  # Duració actual de la fase
        self.phase_cycles = 0              # Cicles en la fase actual
        
        # Matrius de transformació precalculades
        self.rotation_matrices = self._precompute_rotation_matrices()
        
        # Monitorització d'estabilitat
        self.stability_metrics = {
            "determinant_history": [],
            "condition_number_history": [],
            "singular_values_history": [],
            "corrections_applied": 0,
            "stability_warnings": 0
        }
        
        # Estadístiques
        self.stats = {
            "total_phase_transitions": 0,
            "avg_phase_duration": 0,
            "max_angular_velocity": 0,
            "torsion_applications": 0,
            "svd_corrections": 0,
            "stability_rescues": 0
        }
    
    def _precompute_rotation_matrices(self) -> Dict[float, np.ndarray]:
        """Precalcula matrius de rotació per a angles freqüents."""
        matrices = {}
        
        for degrees in range(0, 360, 15):
            radians = math.radians(degrees)
            matrices[degrees] = self._create_rotation_matrix_z(radians)
        
        return matrices
    
    def _create_rotation_matrix_z(self, angle: float) -> np.ndarray:
        """Crea matriu de rotació al voltant de l'eix Z."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        return np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def stabilize_torsion_tensor(self, adaptive: bool = True) -> Dict[str, Any]:
        """
        Garanteix que el tensor de torsió no col·lapsi (Singularity Avoidance).
        Implementa normalització simplèctica via SVD.
        
        Args:
            adaptive: Utilitzar correcció adaptativa basada en estabilitat
            
        Returns:
            Informe de l'estabilització
        """
        T = self.state.torsion_tensor.copy()
        
        # Verificar determinat inicial
        det_before = np.linalg.det(T)
        cond_before = np.linalg.cond(T) if np.linalg.matrix_rank(T) == 3 else float('inf')
        
        correction_report = {
            "correction_applied": False,
            "det_before": det_before,
            "cond_before": cond_before,
            "singular_values_before": [],
            "singular_values_after": [],
            "stability_change": 0.0,
            "reason": "no_correction_needed"
        }
        
        # Verificar necessitat de correcció
        needs_correction = self._check_tensor_stability(T)
        
        if not needs_correction["needs_correction"]:
            # Actualitzar mètriques de seguiment
            _, s, _ = np.linalg.svd(T)
            self.stability_metrics["singular_values_history"].append(s.tolist())
            self.stability_metrics["determinant_history"].append(det_before)
            self.state.structural_stability = min(1.0, self.state.structural_stability * 1.01)
            return correction_report
        
        # 1. Descomposició en Valors Singulars
        try:
            U, s, Vh = np.linalg.svd(T)
            correction_report["singular_values_before"] = s.tolist()
        except np.linalg.LinAlgError:
            # Fallback: reiniciar a identitat
            self.state.torsion_tensor = np.eye(3, dtype=np.float32)
            self.state.structural_stability *= 0.8  # Penalització per error
            
            correction_report.update({
                "correction_applied": True,
                "reason": "svd_failed_reset_to_identity",
                "stability_change": -0.2
            })
            
            self.stats["stability_rescues"] += 1
            return correction_report
        
        # 2. Correcció adaptativa basada en estabilitat estructural
        if adaptive and self.config["adaptive_correction"]:
            stability_factor = self.state.structural_stability
            min_sv = self.config["min_singular_value"] * (0.5 + 0.5 * stability_factor)
        else:
            min_sv = self.config["min_singular_value"]
        
        # 3. Evitar valors singulars nuls o negatius
        s_corrected = np.maximum(s, min_sv)
        
        # 4. Preservar el volum (determinant unitari) si està configurat
        if self.config["preserve_volume"]:
            det_s = np.prod(s_corrected)
            
            if det_s > 0:
                # Escalar per obtenir determinant = 1
                scaling_factor = det_s ** (-1/3)
                s_corrected = s_corrected * scaling_factor
                
                # Verificar que l'escalat no violi el llindar mínim
                if np.any(s_corrected < min_sv):
                    # Ajustar preservant l'ordre relatiu
                    min_ratio = min_sv / np.min(s_corrected)
                    s_corrected = s_corrected * min_ratio
            else:
                # Determinant negatiu o zero, usar valors per defecte
                s_corrected = np.array([1.0, 1.0, 1.0])
        
        # 5. Verificar canvi significatiu
        s_change = np.linalg.norm(s_corrected - s) / np.linalg.norm(s)
        
        if s_change > 0.01:  # Canvi significatiu (més de 1%)
            correction_report["correction_applied"] = True
            correction_report["reason"] = f"singular_values_corrected_change_{s_change:.3f}"
            
            # 6. Reconstrucció del tensor estabilitzat
            T_corrected = U @ np.diag(s_corrected) @ Vh
            
            # 7. Suavitzar la transició (evitar canvis bruscos)
            alpha = 0.3  # Factor de suavitzat
            self.state.torsion_tensor = alpha * T_corrected + (1 - alpha) * T
            
            # Actualitzar estabilitat estructural
            improvement = 1.0 / (1.0 + s_change)  # Millora inversament proporcional al canvi
            self.state.structural_stability = min(1.0, 
                self.state.structural_stability * (1.0 + self.config["stability_recovery_rate"] * improvement))
            
            self.stats["svd_corrections"] += 1
        else:
            # Canvi insignificant, mantenir tensor original
            self.state.torsion_tensor = T
            self.state.structural_stability = min(1.0, 
                self.state.structural_stability * (1.0 + 0.01))  # Petita millora
        
        # Actualitzar mètriques de seguiment
        det_after = np.linalg.det(self.state.torsion_tensor)
        cond_after = np.linalg.cond(self.state.torsion_tensor)
        
        correction_report.update({
            "det_after": det_after,
            "cond_after": cond_after,
            "singular_values_after": s_corrected.tolist(),
            "stability_change": self.state.structural_stability - (det_before / det_after if det_after != 0 else 0),
            "structural_stability": self.state.structural_stability
        })
        
        self.stability_metrics["corrections_applied"] += 1
        self.stability_metrics["determinant_history"].append(det_after)
        self.stability_metrics["condition_number_history"].append(cond_after)
        self.stability_metrics["singular_values_history"].append(s_corrected.tolist())
        
        # Mantenir històrics manejables
        for key in ["determinant_history", "condition_number_history", "singular_values_history"]:
            if len(self.stability_metrics[key]) > 1000:
                self.stability_metrics[key] = self.stability_metrics[key][-1000:]
        
        return correction_report
    
    def _check_tensor_stability(self, T: np.ndarray) -> Dict[str, Any]:
        """
        Verifica l'estabilitat del tensor de torsió.
        
        Returns:
            Dict amb resultats de l'anàlisi d'estabilitat
        """
        stability_check = {
            "needs_correction": False,
            "reasons": [],
            "metrics": {}
        }
        
        try:
            # 1. Verificar determinat
            det = np.linalg.det(T)
            stability_check["metrics"]["determinant"] = det
            
            if abs(det) < self.config["min_singular_value"] ** 3:
                stability_check["needs_correction"] = True
                stability_check["reasons"].append(f"determinant_too_small: {det:.2e}")
            
            # 2. Verificar nombre de condició
            cond = np.linalg.cond(T)
            stability_check["metrics"]["condition_number"] = cond
            
            if cond > 1e8:  # Condició numèrica pobra
                stability_check["needs_correction"] = True
                stability_check["reasons"].append(f"condition_number_too_high: {cond:.2e}")
            
            # 3. Verificar valors singulars
            _, s, _ = np.linalg.svd(T)
            stability_check["metrics"]["singular_values"] = s.tolist()
            
            min_sv = np.min(s)
            max_sv = np.max(s)
            sv_ratio = max_sv / (min_sv + 1e-10)
            
            if min_sv < self.config["min_singular_value"]:
                stability_check["needs_correction"] = True
                stability_check["reasons"].append(f"min_singular_value_too_small: {min_sv:.2e}")
            
            if sv_ratio > 1e6:
                stability_check["needs_correction"] = True
                stability_check["reasons"].append(f"singular_value_ratio_too_high: {sv_ratio:.2e}")
            
            # 4. Verificar simetria (per a estabilitat geomètrica)
            asymmetry = np.linalg.norm(T - T.T) / np.linalg.norm(T)
            stability_check["metrics"]["asymmetry"] = asymmetry
            
            if asymmetry > 10.0:  # Tensor molt asimètric
                stability_check["needs_correction"] = True
                stability_check["reasons"].append(f"high_asymmetry: {asymmetry:.2f}")
            
        except np.linalg.LinAlgError:
            # Error en càlculs lineals → correcció necessària
            stability_check["needs_correction"] = True
            stability_check["reasons"].append("linear_algebra_error")
            stability_check["metrics"] = {"error": "LinAlgError"}
        
        return stability_check
    
    def update_angular_state(self, 
                           position: np.ndarray,
                           linear_momentum: np.ndarray) -> AngularState:
        """
        Actualitza l'estat angular basant-se en la posició i momentum.
        
        Args:
            position: Posició actual
            linear_momentum: Momentum lineal
            
        Returns:
            Nou estat angular
        """
        # Incrementar comptador de cicles
        self.phase_cycles += 1
        
        # PAS 1: Calcular momentum angular
        new_angular_momentum = self._calculate_angular_momentum(
            position, linear_momentum
        )
        
        # PAS 2: Aplicar amortiment
        self.state.angular_momentum = (
            new_angular_momentum * self.config["angular_damping"]
        )
        
        # PAS 3: Limitar velocitat angular
        angular_velocity = np.linalg.norm(self.state.angular_momentum)
        self.stats["max_angular_velocity"] = max(
            self.stats["max_angular_velocity"], angular_velocity
        )
        
        if angular_velocity > self.config["max_angular_velocity"]:
            scale_factor = self.config["max_angular_velocity"] / angular_velocity
            self.state.angular_momentum *= scale_factor
        
        # PAS 4: Actualitzar tensor de torsió
        self._update_torsion_tensor()
        
        # PAS 5: Aplicar estabilització SVD (periòdica o quan calgui)
        if (self.config["svd_correction_enabled"] and 
            (self.phase_cycles % self.config["stability_check_interval"] == 0 or 
             self.state.structural_stability < 0.7)):
            
            correction_report = self.stabilize_torsion_tensor(
                adaptive=self.config["adaptive_correction"]
            )
            
            if correction_report["correction_applied"]:
                # Registrar correcció aplicada
                if self.state.phase == AngularPhase.TORSION:
                    # En fase de torsió, acceptar més canvis
                    self.state.structural_stability = max(0.3, 
                        self.state.structural_stability * 0.95)
                else:
                    # En altres fases, mantenir alta estabilitat
                    self.state.structural_stability = max(0.7,
                        self.state.structural_stability * 0.98)
        
        # PAS 6: Aplicar decaïment estructural natural
        self.state.structural_stability *= self.config["structural_decay_rate"]
        
        # PAS 7: Actualitzar progrés de fase
        self.state.phase_progress = min(1.0, self.phase_cycles / self.current_phase_duration)
        
        # PAS 8: Calcular entropia angular
        self.state.angular_entropy = self._calculate_angular_entropy()
        
        # PAS 9: Verificar transició de fase
        should_transition = self._should_transition_phase()
        
        if should_transition:
            self._transition_to_next_phase()
        
        return self.state
    
    def _calculate_angular_momentum(self, 
                                  position: np.ndarray, 
                                  linear_momentum: np.ndarray) -> np.ndarray:
        """Calcula momentum angular a partir de posició i momentum lineal."""
        
        # Momentum angular = r × p (producte vectorial)
        if np.linalg.norm(position) > 0 and np.linalg.norm(linear_momentum) > 0:
            angular_momentum = np.cross(position, linear_momentum)
        else:
            # Valor petit aleatori per evitar estancament
            angular_momentum = np.random.normal(0, 0.01, 3)
        
        return angular_momentum
    
    def _update_torsion_tensor(self):
        """Actualitza el tensor de torsió basant-se en l'estat actual."""
        
        # Base del tensor (identitat amb pertorbació)
        base_tensor = np.eye(3, dtype=np.float32)
        
        # Factor basat en estabilitat estructural
        stability_factor = self.state.structural_stability
        
        # Afegir components basats en momentum angular
        angular_norm = np.linalg.norm(self.state.angular_momentum)
        
        if angular_norm > 0:
            # Normalitzar momentum angular
            angular_dir = self.state.angular_momentum / angular_norm
            
            # Força de torsió ajustada per estabilitat
            torsion_strength = min(self.config["torsion_strength"] * stability_factor, 
                                 angular_norm * 0.1)
            
            # Crear matriu de torsió (producte exterior)
            torsion_component = np.outer(angular_dir, angular_dir) * torsion_strength
            
            # Afegir component antisimètrica (rotació)
            skew_symmetric = self._create_skew_symmetric(
                angular_dir * torsion_strength * 0.5 * stability_factor
            )
            
            # Combinar components
            T_new = base_tensor + torsion_component + skew_symmetric
            
            # Suavitzar la transició
            alpha = 0.2 + 0.6 * stability_factor  # Menys suavitzat amb alta estabilitat
            self.state.torsion_tensor = alpha * T_new + (1 - alpha) * self.state.torsion_tensor
            
            self.stats["torsion_applications"] += 1
        else:
            # Tensor identitat amb petita pertorbació aleatòria controlada
            perturbation_strength = 0.01 * (1.0 - stability_factor)  # Menys pertorbació amb alta estabilitat
            perturbation = np.random.normal(0, perturbation_strength, (3, 3))
            self.state.torsion_tensor = base_tensor + perturbation
    
    def _create_skew_symmetric(self, vector: np.ndarray) -> np.ndarray:
        """Crea una matriu antisimètrica a partir d'un vector."""
        return np.array([
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0]
        ], dtype=np.float32)
    
    def _calculate_angular_entropy(self) -> float:
        """Calcula l'entropia angular (mesura de caos/ordre)."""
        
        # Utilitzar valors propis del tensor de torsió
        eigenvalues = np.linalg.eigvals(self.state.torsion_tensor)
        
        if len(eigenvalues) < 2:
            return 0.5
        
        # Normalitzar valors propis
        abs_eigenvalues = np.abs(eigenvalues)
        normalized = abs_eigenvalues / (np.sum(abs_eigenvalues) + 1e-10)
        
        # Entropia de Shannon
        entropy = -np.sum(normalized * np.log2(normalized + 1e-10))
        
        # Normalitzar entre 0 i 1
        max_entropy = np.log2(len(eigenvalues))
        normalized_entropy = entropy / max_entropy
        
        # Modificar per estabilitat estructural
        stability_modulation = 1.0 - 0.3 * (1.0 - self.state.structural_stability)
        
        return float(normalized_entropy * stability_modulation)
    
    def _should_transition_phase(self) -> bool:
        """Determina si s'ha de produir una transició de fase."""
        
        # Condicions per a transició:
        # 1. Ha passat el temps mínim
        if self.phase_cycles < 10:
            return False
        
        # 2. S'ha completat la fase actual
        if self.state.phase_progress >= 1.0:
            return True
        
        # 3. Alta entropia angular (caos)
        if self.state.angular_entropy > self.config["entropy_threshold"]:
            return True
        
        # 4. Momentum angular molt baix (estancament)
        angular_norm = np.linalg.norm(self.state.angular_momentum)
        if angular_norm < 0.01 and self.phase_cycles > 30:
            return True
        
        # 5. Baixa estabilitat estructural (necessitat de reinici)
        if self.state.structural_stability < 0.3:
            return True
        
        return False
    
    def _transition_to_next_phase(self):
        """Realitza una transició a la següent fase angular."""
        
        # Registrar fase actual en l'historial
        phase_record = {
            "phase": self.state.phase.value,
            "duration_cycles": self.phase_cycles,
            "final_entropy": self.state.angular_entropy,
            "structural_stability": self.state.structural_stability,
            "determinant": float(np.linalg.det(self.state.torsion_tensor)),
            "timestamp": datetime.now().isoformat()
        }
        self.phase_history.append(phase_record)
        
        # Determinar següent fase (cicle predefinit)
        phase_sequence = [
            AngularPhase.EMERGENCE,
            AngularPhase.COHERENCE,
            AngularPhase.TORSION,
            AngularPhase.INTEGRATION,
            AngularPhase.TRANSITION
        ]
        
        current_idx = phase_sequence.index(self.state.phase)
        next_idx = (current_idx + 1) % len(phase_sequence)
        
        # Actualitzar fase
        old_phase = self.state.phase
        self.state.phase = phase_sequence[next_idx]
        self.state.last_phase_transition = datetime.now()
        
        # Reiniciar comptadors
        self.phase_cycles = 0
        
        # Seleccionar nova durada de fase
        min_dur, max_dur = self.config["phase_duration_range"]
        self.current_phase_duration = np.random.randint(min_dur, max_dur)
        
        # Reiniciar progrés
        self.state.phase_progress = 0.0
        
        # Ajustar tensor de torsió per a la nova fase
        self._adjust_torsion_for_phase(old_phase)
        
        # Actualitzar estadístiques
        self.stats["total_phase_transitions"] += 1
        
        # Calcular durada mitjana de fase
        if len(self.phase_history) > 0:
            durations = [p["duration_cycles"] for p in self.phase_history]
            self.stats["avg_phase_duration"] = np.mean(durations)
    
    def _adjust_torsion_for_phase(self, old_phase: AngularPhase):
        """Ajusta el tensor de torsió per a la nova fase."""
        
        if self.state.phase == AngularPhase.EMERGENCE:
            # Torsió suau per a emergència
            self.state.torsion_tensor = np.eye(3) * 0.8 + np.random.normal(0, 0.05, (3, 3))
            self.state.structural_stability = 0.9  # Alta estabilitat inicial
        
        elif self.state.phase == AngularPhase.COHERENCE:
            # Tensor més estable, menys torsió
            self.state.torsion_tensor = np.eye(3) * 0.9 + np.random.normal(0, 0.02, (3, 3))
            self.state.structural_stability = 0.95
        
        elif self.state.phase == AngularPhase.TORSION:
            # Augmentar torsió significativament
            torsion_strength = self.config["torsion_strength"] * 1.5
            random_torsion = np.random.normal(0, torsion_strength, (3, 3))
            self.state.torsion_tensor = np.eye(3) + random_torsion
            self.state.structural_stability = 0.7  # Acceptar menor estabilitat
        
        elif self.state.phase == AngularPhase.INTEGRATION:
            # Reduir torsió gradualment
            self.state.torsion_tensor = self.state.torsion_tensor * 0.7 + np.eye(3) * 0.3
            self.state.structural_stability = min(1.0, 
                self.state.structural_stability * 1.1)  # Recuperar estabilitat
        
        elif self.state.phase == AngularPhase.TRANSITION:
            # Torsió intermèdia per a transició
            self.state.torsion_tensor = (self.state.torsion_tensor + np.eye(3)) / 2
            self.state.structural_stability = 0.85
        
        # Aplicar estabilització després del canvi de fase
        if self.config["svd_correction_enabled"]:
            self.stabilize_torsion_tensor(adaptive=True)
    
    def apply_angular_transformation(self, 
                                   points: np.ndarray) -> np.ndarray:
        """
        Aplica la transformació angular actual a un conjunt de punts.
        
        Args:
            points: Array de punts 3D (N×3)
            
        Returns:
            Punts transformats
        """
        if len(points.shape) == 1:
            points = points.reshape(1, -1)
        
        # Verificar estabilitat abans d'aplicar
        stability_check = self._check_tensor_stability(self.state.torsion_tensor)
        
        if stability_check["needs_correction"]:
            # Aplicar correcció urgent
            self.stabilize_torsion_tensor(adaptive=False)
        
        # Aplicar tensor de torsió
        transformed_points = np.dot(points, self.state.torsion_tensor.T)
        
        return transformed_points
    
    def generate_angular_svg_transform(self) -> str:
        """
        Genera una cadena de transformació SVG basada en l'estat angular.
        
        Returns:
            Cadena de transformació CSS/SVG
        """
        # Verificar estabilitat del tensor
        stability_check = self._check_tensor_stability(self.state.torsion_tensor)
        
        if stability_check["needs_correction"]:
            # Tensor inestable, generar transformació segura
            safe_transform = self._generate_safe_svg_transform()
            return safe_transform
        
        # Extraure components de rotació del tensor
        rotation_angle = self._extract_rotation_angle()
        scale_factors = self._extract_scale_factors()
        
        # Construir transformació CSS
        transforms = []
        
        # Rotació (en graus)
        transforms.append(f"rotate({rotation_angle:.1f})")
        
        # Escalat
        scale_x, scale_y, _ = scale_factors
        transforms.append(f"scale({scale_x:.3f}, {scale_y:.3f})")
        
        # Inclinar basant-se en components no diagonals
        skew_x, skew_y = self._extract_skew_angles()
        if abs(skew_x) > 0.1 or abs(skew_y) > 0.1:
            transforms.append(f"skewX({skew_x:.1f})")
            transforms.append(f"skewY({skew_y:.1f})")
        
        # Afegir transició suau si no estem en fase de torsió
        if self.state.phase != AngularPhase.TORSION:
            transition = "transition: transform 0.3s ease;"
        else:
            transition = ""
        
        transform_string = " ".join(transforms)
        
        return f"transform: {transform_string}; {transition}"
    
    def _generate_safe_svg_transform(self) -> str:
        """Genera una transformació SVG segura quan el tensor és inestable."""
        
        # Transformació identitat amb petita aleatorietat
        small_angle = np.random.uniform(-5, 5)
        small_scale = 0.95 + np.random.uniform(-0.05, 0.05)
        
        return f"transform: rotate({small_angle:.1f}) scale({small_scale:.3f}); transition: transform 0.5s ease;"
    
    def _extract_rotation_angle(self) -> float:
        """Extrau angle de rotació del tensor de torsió."""
        # Utilitzar component 2D (XY) per a rotació SVG
        tensor_2d = self.state.torsion_tensor[:2, :2]
        
        # Calcular angle a partir de la matriu de rotació
        if np.linalg.det(tensor_2d) > 0:
            # És una rotació pròpia
            cos_theta = (tensor_2d[0, 0] + tensor_2d[1, 1]) / 2
            sin_theta = (tensor_2d[1, 0] - tensor_2d[0, 1]) / 2
            angle = math.atan2(sin_theta, cos_theta)
        else:
            # Inclou reflexió, usar un valor més simple
            angle = math.atan2(tensor_2d[1, 0], tensor_2d[0, 0])
        
        return math.degrees(angle)
    
    def _extract_scale_factors(self) -> Tuple[float, float, float]:
        """Extrau factors d'escala del tensor de torsió."""
        # Els valors singulars donen els factors d'escala
        u, s, vh = np.linalg.svd(self.state.torsion_tensor)
        
        return (s[0], s[1], s[2])
    
    def _extract_skew_angles(self) -> Tuple[float, float]:
        """Extrau angles d'inclinació del tensor de torsió."""
        tensor_2d = self.state.torsion_tensor[:2, :2]
        
        # Descomposició QR per separar rotació i inclinació
        q, r = np.linalg.qr(tensor_2d)
        
        # Els elements no diagonals de R donen la inclinació
        if abs(r[0, 0]) > 1e-10:
            skew_y = math.atan2(r[0, 1], r[0, 0])
        else:
            skew_y = 0.0
        
        if abs(r[1, 1]) > 1e-10:
            skew_x = math.atan2(r[1, 0], r[1, 1])
        else:
            skew_x = 0.0
        
        return math.degrees(skew_x), math.degrees(skew_y)
    
    def get_angular_report(self) -> Dict[str, Any]:
        """Genera informe de l'estat angular."""
        
        # Calcular métriques actuals
        angular_velocity = float(np.linalg.norm(self.state.angular_momentum))
        torsion_magnitude = float(np.linalg.norm(self.state.torsion_tensor - np.eye(3)))
        
        # Verificar estabilitat
        stability_check = self._check_tensor_stability(self.state.torsion_tensor)
        det = float(np.linalg.det(self.state.torsion_tensor))
        
        # Distribució de temps per fase
        phase_durations = {}
        for record in self.phase_history[-10:]:  # Últimes 10 fases
            phase = record["phase"]
            phase_durations[phase] = phase_durations.get(phase, 0) + record["duration_cycles"]
        
        # Històric recent d'estabilitat
        recent_determinants = self.stability_metrics["determinant_history"][-20:] if self.stability_metrics["determinant_history"] else []
        recent_conditions = self.stability_metrics["condition_number_history"][-20:] if self.stability_metrics["condition_number_history"] else []
        
        return {
            "angular_state": {
                "phase": self.state.phase.value,
                "phase_progress": self.state.phase_progress,
                "angular_entropy": self.state.angular_entropy,
                "angular_velocity": angular_velocity,
                "torsion_magnitude": torsion_magnitude,
                "structural_stability": self.state.structural_stability,
                "cycles_in_current_phase": self.phase_cycles,
                "determinant": det,
                "condition_number": stability_check.get("metrics", {}).get("condition_number", 0)
            },
            "stability_analysis": {
                "needs_correction": stability_check["needs_correction"],
                "correction_reasons": stability_check["reasons"],
                "corrections_applied": self.stats["svd_corrections"],
                "stability_rescues": self.stats["stability_rescues"],
                "recent_determinants": recent_determinants,
                "recent_condition_numbers": recent_conditions
            },
            "phase_statistics": {
                "total_transitions": self.stats["total_phase_transitions"],
                "avg_phase_duration": self.stats["avg_phase_duration"],
                "max_angular_velocity": self.stats["max_angular_velocity"],
                "recent_phase_durations": phase_durations
            },
            "transformation_info": {
                "svg_transform": self.generate_angular_svg_transform(),
                "rotation_angle": self._extract_rotation_angle(),
                "scale_factors": self._extract_scale_factors(),
                "tensor_stable": not stability_check["needs_correction"]
            }
        }
    
    def reset_angular_state(self):
        """Reinicia l'estat angular a valors inicials."""
        self.state = AngularState(
            phase=AngularPhase.EMERGENCE,
            angular_momentum=np.zeros(3),
            torsion_tensor=np.eye(3),
            phase_progress=0.0,
            angular_entropy=0.5,
            structural_stability=1.0
        )
        
        self.phase_cycles = 0
        self.current_phase_duration = 100
        
        # Reiniciar mètriques d'estabilitat
        self.stability_metrics = {
            "determinant_history": [],
            "condition_number_history": [],
            "singular_values_history": [],
            "corrections_applied": 0,
            "stability_warnings": 0
        }
        
        print("Estat angular reiniciat (amb estabilització SVD)")
    
    def force_stability_correction(self) -> Dict[str, Any]:
        """
        Força una correcció d'estabilitat (per a recuperació d'errors).
        
        Returns:
            Informe de la correcció forçada
        """
        print("Forçant correcció d'estabilitat...")
        
        # Aplicar correcció agressiva
        old_tensor = self.state.torsion_tensor.copy()
        old_stability = self.state.structural_stability
        
        # Estabilitzar tensor
        correction_report = self.stabilize_torsion_tensor(adaptive=False)
        
        # Millorar estabilitat estructural
        self.state.structural_stability = max(0.8, old_stability * 1.2)
        
        # Calcular canvi
        tensor_change = np.linalg.norm(self.state.torsion_tensor - old_tensor) / np.linalg.norm(old_tensor)
        
        correction_report.update({
            "forced_correction": True,
            "tensor_change": tensor_change,
            "structural_stability_before": old_stability,
            "structural_stability_after": self.state.structural_stability
        })
        
        return correction_report