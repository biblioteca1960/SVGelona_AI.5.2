"""
SVGelona_AI 5.2 - Transformador CSS Matricial
Conversió de tensors a transformacions CSS optimitzades per GPU.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import math

@dataclass
class CSSTransform:
    """Transformació CSS amb múltiples components."""
    
    matrix_3d: Optional[str] = None
    matrix_2d: Optional[str] = None
    translate_3d: Optional[str] = None
    rotate_3d: Optional[str] = None
    scale_3d: Optional[str] = None
    skew: Optional[str] = None
    perspective: Optional[str] = None
    
    def to_css_string(self, use_3d: bool = True, optimize: bool = True) -> str:
        """
        Converteix a cadena CSS.
        
        Args:
            use_3d: Utilitzar transformacions 3D si estan disponibles
            optimize: Optimitzar per a rendiment GPU
            
        Returns:
            Cadena de transformació CSS
        """
        transforms = []
        
        if optimize and use_3d and self.matrix_3d:
            # matrix3d() és la més eficient per GPU
            transforms.append(f"matrix3d({self.matrix_3d})")
        
        elif not optimize and use_3d:
            # Descomposició explícita (més llegible)
            if self.translate_3d:
                transforms.append(f"translate3d({self.translate_3d})")
            if self.rotate_3d:
                transforms.append(f"rotate3d({self.rotate_3d})")
            if self.scale_3d:
                transforms.append(f"scale3d({self.scale_3d})")
            if self.perspective:
                transforms.append(f"perspective({self.perspective})")
        
        else:
            # Mode 2D
            if self.matrix_2d:
                transforms.append(f"matrix({self.matrix_2d})")
            elif self.skew:
                transforms.append(f"skew({self.skew})")
        
        if not transforms:
            return "transform: none;"
        
        transform_string = " ".join(transforms)
        return f"transform: {transform_string};"
    
    def to_svg_transform(self) -> str:
        """Converteix a cadena de transformació SVG."""
        if self.matrix_2d:
            # SVG utilitza matrix(a, b, c, d, e, f)
            return f"matrix({self.matrix_2d})"
        elif self.matrix_3d:
            # SVG no suporta 3D complet, convertir a 2D
            matrix_2d = self._extract_2d_from_3d()
            return f"matrix({matrix_2d})"
        else:
            return ""

    def _extract_2d_from_3d(self) -> str:
        """Extreu components 2D d'una matriu 3D."""
        if not self.matrix_3d:
            return "1, 0, 0, 1, 0, 0"
        
        # Parsejar matriu 3D
        values = [float(v.strip()) for v in self.matrix_3d.split(',')]
        if len(values) != 16:
            return "1, 0, 0, 1, 0, 0"
        
        # Extraure components 2D (primera submatriu 2x2 + translació)
        a, b, _, _ = values[0:4]
        c, d, _, _ = values[4:8]
        e, f, _, _ = values[12:16]
        
        return f"{a}, {b}, {c}, {d}, {e}, {f}"

class CSSMatrixTransformer:
    """
    Transformador que converteix tensors i estats angulars a CSS optimitzat.
    """
    
    def __init__(self):
        # Cache de transformacions
        self.transform_cache: Dict[Tuple, CSSTransform] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Configuració d'optimització
        self.config = {
            "use_gpu_acceleration": True,
            "optimize_for_performance": True,
            "max_cache_size": 1000,
            "precision_digits": 6,
            "simplify_matrices": True,
            "compress_transforms": True
        }
        
        # Precomputar matrius comunes
        self.common_matrices = self._precompute_common_matrices()
        
        # Estadístiques
        self.stats = {
            "transforms_generated": 0,
            "gpu_optimized": 0,
            "cache_efficiency": 0.0
        }
    
    def _precompute_common_matrices(self) -> Dict[str, np.ndarray]:
        """Precomputa matrius de transformació comunes."""
        matrices = {}
        
        # Identitat
        matrices["identity"] = np.eye(4, dtype=np.float32)
        
        # Rotacions comunes
        for angle in [0, 30, 45, 60, 90, 180, 270]:
            rad = math.radians(angle)
            matrices[f"rotate_x_{angle}"] = self._rotation_matrix_x(rad)
            matrices[f"rotate_y_{angle}"] = self._rotation_matrix_y(rad)
            matrices[f"rotate_z_{angle}"] = self._rotation_matrix_z(rad)
        
        # Escalats comuns
        for scale in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            matrices[f"scale_{scale}"] = np.diag([scale, scale, scale, 1])
        
        return matrices
    
    def _rotation_matrix_x(self, angle: float) -> np.ndarray:
        """Matriu de rotació al voltant de l'eix X."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        return np.array([
            [1, 0, 0, 0],
            [0, cos_a, -sin_a, 0],
            [0, sin_a, cos_a, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
    
    def _rotation_matrix_y(self, angle: float) -> np.ndarray:
        """Matriu de rotació al voltant de l'eix Y."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        return np.array([
            [cos_a, 0, sin_a, 0],
            [0, 1, 0, 0],
            [-sin_a, 0, cos_a, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
    
    def _rotation_matrix_z(self, angle: float) -> np.ndarray:
        """Matriu de rotació al voltant de l'eix Z."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        return np.array([
            [cos_a, -sin_a, 0, 0],
            [sin_a, cos_a, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
    
    def torsion_tensor_to_css_matrix(self,
                                    torsion_tensor: np.ndarray,
                                    position: np.ndarray,
                                    phase: str = "coherence") -> CSSTransform:
        """
        Converteix un tensor de torsió a transformació CSS.
        
        Args:
            torsion_tensor: Tensor de torsió 3x3
            position: Posició actual (per a perspectiva)
            phase: Fase angular (per a optimitzacions)
            
        Returns:
            Objecte CSSTransform
        """
        self.stats["transforms_generated"] += 1
        
        # Crear clau de cache
        cache_key = self._create_cache_key(torsion_tensor, position, phase)
        
        # Verificar cache
        if cache_key in self.transform_cache:
            self.cache_hits += 1
            return self.transform_cache[cache_key]
        
        self.cache_misses += 1
        
        # Convertir tensor 3x3 a matriu de transformació 4x4
        transform_matrix = self._tensor_to_transform_matrix(torsion_tensor, position)
        
        # Optimitzar segons fase
        if phase in ["torsion", "transition"]:
            # Fases amb alta activitat → optimitzar per GPU
            css_transform = self._create_gpu_optimized_transform(transform_matrix)
            self.stats["gpu_optimized"] += 1
        else:
            # Fases estables → transformació més simple
            css_transform = self._create_simplified_transform(transform_matrix)
        
        # Simplificar si està configurat
        if self.config["simplify_matrices"]:
            css_transform = self._simplify_transform(css_transform)
        
        # Comprimir si està configurat
        if self.config["compress_transforms"]:
            css_transform = self._compress_transform(css_transform)
        
        # Actualitzar cache
        if len(self.transform_cache) < self.config["max_cache_size"]:
            self.transform_cache[cache_key] = css_transform
        
        # Actualitzar estadístiques de cache
        total_accesses = self.cache_hits + self.cache_misses
        if total_accesses > 0:
            self.stats["cache_efficiency"] = self.cache_hits / total_accesses
        
        return css_transform
    
    def _create_cache_key(self,
                         torsion_tensor: np.ndarray,
                         position: np.ndarray,
                         phase: str) -> Tuple:
        """Crea una clau de cache per a una transformació."""
        
        # Arrodonir valors per a agrupar transformacions similars
        tensor_key = tuple(np.round(torsion_tensor.flatten(), 3))
        position_key = tuple(np.round(position, 2))
        
        return (tensor_key, position_key, phase)
    
    def _tensor_to_transform_matrix(self,
                                   torsion_tensor: np.ndarray,
                                   position: np.ndarray) -> np.ndarray:
        """Converteix un tensor de torsió a matriu de transformació 4x4."""
        
        # Expandir tensor 3x3 a 4x4
        transform_matrix = np.eye(4, dtype=np.float32)
        transform_matrix[:3, :3] = torsion_tensor
        
        # Afegir translació basada en la posició
        translation_scale = 0.1  # Escalar translació per a visualització
        transform_matrix[:3, 3] = position * translation_scale
        
        # Afegir perspectiva basada en la distància a l'origen
        distance = np.linalg.norm(position)
        if distance > 0:
            perspective_value = 1000 / (1 + distance)  # Perspectiva inversament proporcional
            transform_matrix[3, 2] = -1 / perspective_value  # Component de perspectiva
        
        return transform_matrix
    
    def _create_gpu_optimized_transform(self, transform_matrix: np.ndarray) -> CSSTransform:
        """Crea una transformació optimitzada per a GPU."""
        
        css_transform = CSSTransform()
        
        # matrix3d() és la més eficient per GPU
        matrix_3d_string = self._matrix_to_css_string(transform_matrix)
        css_transform.matrix_3d = matrix_3d_string
        
        # També proporcionar descomposició per a compatibilitat
        if not self.config["optimize_for_performance"]:
            # Calcular components descompostos
            translation, rotation, scale = self._decompose_transform(transform_matrix)
            
            css_transform.translate_3d = f"{translation[0]:.2f}px, {translation[1]:.2f}px, {translation[2]:.2f}px"
            css_transform.rotate_3d = f"{rotation[0]:.2f}, {rotation[1]:.2f}, {rotation[2]:.2f}, {rotation[3]:.1f}deg"
            css_transform.scale_3d = f"{scale[0]:.2f}, {scale[1]:.2f}, {scale[2]:.2f}"
        
        return css_transform
    
    def _create_simplified_transform(self, transform_matrix: np.ndarray) -> CSSTransform:
        """Crea una transformació simplificada (2D o descomposta)."""
        
        css_transform = CSSTransform()
        
        # Extraure submatriu 2D
        matrix_2d = transform_matrix[:2, :2]
        translation_2d = transform_matrix[:2, 3]
        
        # Crear cadena matrix() 2D
        a, b = matrix_2d[0]
        c, d = matrix_2d[1]
        e, f = translation_2d
        
        matrix_2d_string = f"{a:.{self.config['precision_digits']}f}, {b:.{self.config['precision_digits']}f}, " \
                          f"{c:.{self.config['precision_digits']}f}, {d:.{self.config['precision_digits']}f}, " \
                          f"{e:.{self.config['precision_digits']}f}, {f:.{self.config['precision_digits']}f}"
        
        css_transform.matrix_2d = matrix_2d_string
        
        # Calcular skew si és significatiu
        skew_x, skew_y = self._calculate_skew(matrix_2d)
        if abs(skew_x) > 0.1 or abs(skew_y) > 0.1:
            css_transform.skew = f"{skew_x:.1f}deg, {skew_y:.1f}deg"
        
        return css_transform
    
    def _matrix_to_css_string(self, matrix: np.ndarray) -> str:
        """Converteix una matriu 4x4 a cadena CSS matrix3d()."""
        
        # Aplanar matriu en ordre de fila major
        flat_matrix = matrix.flatten(order='C')
        
        # Formatar valors
        formatted_values = []
        for value in flat_matrix:
            # Arrodonir per a reduir mida
            if abs(value) < 1e-10:
                formatted_value = "0"
            else:
                formatted_value = f"{value:.{self.config['precision_digits']}f}"
            formatted_values.append(formatted_value)
        
        return ", ".join(formatted_values)
    
    def _decompose_transform(self, transform_matrix: np.ndarray) -> Tuple:
        """
        Descomposa una matriu de transformació en translació, rotació i escala.
        
        Returns:
            Tupla (translation, rotation, scale)
        """
        # Extraure translació (última columna, excluint component homogènia)
        translation = transform_matrix[:3, 3].copy()
        
        # Extraure submatriu de rotació-escala
        M = transform_matrix[:3, :3].copy()
        
        # Descomposició QR per separar rotació i escala
        Q, R = np.linalg.qr(M)
        
        # Assegurar que Q és una rotació pròpia (det = 1)
        if np.linalg.det(Q) < 0:
            Q = -Q
        
        # Convertir Q a angle-eix
        angle, axis = self._rotation_matrix_to_axis_angle(Q)
        
        # L'escala està a la diagonal de R
        scale = np.array([R[0, 0], R[1, 1], R[2, 2]])
        
        # Assegurar valors positius
        scale = np.abs(scale)
        
        rotation = (*axis, math.degrees(angle))
        
        return translation, rotation, scale
    
    def _rotation_matrix_to_axis_angle(self, R: np.ndarray) -> Tuple:
        """Converteix una matriu de rotació a representació angle-eix."""
        
        # Angle de rotació
        angle = math.acos(min(1.0, max(-1.0, (np.trace(R) - 1) / 2)))
        
        if abs(angle) < 1e-10:
            # Rotació zero, eix arbitrari
            axis = np.array([1.0, 0.0, 0.0])
        else:
            # Eix de rotació
            axis = np.array([
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1]
            ]) / (2 * math.sin(angle))
        
        # Normalitzar eix
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 0:
            axis = axis / axis_norm
        
        return angle, axis
    
    def _calculate_skew(self, matrix_2d: np.ndarray) -> Tuple[float, float]:
        """Calcula angles d'inclinació a partir d'una matriu 2D."""
        
        # Descomposició QR
        Q, R = np.linalg.qr(matrix_2d)
        
        # Angles d'inclinació estan als elements no diagonals de R
        if abs(R[0, 0]) > 1e-10:
            skew_y = math.atan2(R[0, 1], R[0, 0])
        else:
            skew_y = 0.0
        
        if abs(R[1, 1]) > 1e-10:
            skew_x = math.atan2(R[1, 0], R[1, 1])
        else:
            skew_x = 0.0
        
        return math.degrees(skew_x), math.degrees(skew_y)
    
    def _simplify_transform(self, css_transform: CSSTransform) -> CSSTransform:
        """Simplifica una transformació CSS eliminant components insignificant."""
        
        simplified = CSSTransform()
        
        # Prioritzar matrix3d si està disponible
        if css_transform.matrix_3d:
            # Verificar si és gairebé identitat
            if self._is_near_identity(css_transform.matrix_3d):
                # Transformació trivial → retornar buida
                return simplified
            
            simplified.matrix_3d = css_transform.matrix_3d
        
        # Verificar altres components
        if css_transform.matrix_2d and not simplified.matrix_3d:
            simplified.matrix_2d = css_transform.matrix_2d
        
        # Components 3D només si són significatius
        if css_transform.translate_3d:
            if self._is_significant_translation(css_transform.translate_3d):
                simplified.translate_3d = css_transform.translate_3d
        
        if css_transform.rotate_3d:
            if self._is_significant_rotation(css_transform.rotate_3d):
                simplified.rotate_3d = css_transform.rotate_3d
        
        if css_transform.scale_3d:
            if self._is_significant_scale(css_transform.scale_3d):
                simplified.scale_3d = css_transform.scale_3d
        
        return simplified
    
    def _is_near_identity(self, matrix_string: str) -> bool:
        """Verifica si una matriu és gairebé la identitat."""
        
        try:
            values = [float(v.strip()) for v in matrix_string.split(',')]
            if len(values) != 16:
                return False
            
            # Matriu identitat esperada
            identity = np.eye(4).flatten()
            
            # Comparar
            diff = np.abs(np.array(values) - identity)
            max_diff = np.max(diff)
            
            return max_diff < 0.01  # Llindar del 1%
        except:
            return False
    
    def _is_significant_translation(self, translate_string: str) -> bool:
        """Verifica si una translació és significativa."""
        
        try:
            # Parsejar "x, y, z"
            parts = translate_string.replace('px', '').split(',')
            if len(parts) != 3:
                return True  # Conservar si no es pot parsejar
            
            values = [abs(float(p.strip())) for p in parts]
            max_translation = max(values)
            
            return max_translation > 0.1  # Més de 0.1px
        except:
            return True
    
    def _is_significant_rotation(self, rotate_string: str) -> bool:
        """Verifica si una rotació és significativa."""
        
        try:
            # Parsejar "x, y, z, angle"
            parts = rotate_string.replace('deg', '').split(',')
            if len(parts) != 4:
                return True
            
            angle = abs(float(parts[3].strip()))
            return angle > 0.1  # Més de 0.1 graus
        except:
            return True
    
    def _is_significant_scale(self, scale_string: str) -> bool:
        """Verifica si un escalat és significatiu."""
        
        try:
            # Parsejar "x, y, z"
            parts = scale_string.split(',')
            if len(parts) != 3:
                return True
            
            values = [abs(float(p.strip())) for p in parts]
            
            # Verificar desviació de 1.0
            deviations = [abs(v - 1.0) for v in values]
            max_deviation = max(deviations)
            
            return max_deviation > 0.01  # Més de l'1% de desviació
        except:
            return True
    
    def _compress_transform(self, css_transform: CSSTransform) -> CSSTransform:
        """Comprimeix una transformació eliminant decimals innecessaris."""
        
        compressed = CSSTransform()
        
        # Comprimir matrix3d
        if css_transform.matrix_3d:
            compressed.matrix_3d = self._compress_matrix_string(css_transform.matrix_3d)
        
        # Comprimir matrix2d
        if css_transform.matrix_2d:
            compressed.matrix_2d = self._compress_matrix_string(css_transform.matrix_2d)
        
        # Comprimir altres components
        if css_transform.translate_3d:
            compressed.translate_3d = self._compress_translation_string(css_transform.translate_3d)
        
        if css_transform.rotate_3d:
            compressed.rotate_3d = self._compress_rotation_string(css_transform.rotate_3d)
        
        if css_transform.scale_3d:
            compressed.scale_3d = self._compress_scale_string(css_transform.scale_3d)
        
        return compressed
    
    def _compress_matrix_string(self, matrix_string: str) -> str:
        """Comprimeix una cadena de matriu eliminant decimals innecessaris."""
        
        parts = matrix_string.split(',')
        compressed_parts = []
        
        for part in parts:
            try:
                value = float(part.strip())
                
                # Arrodonir segons precisió configurada
                if abs(value) < 1e-10:
                    compressed = "0"
                elif abs(value - 1.0) < 1e-10:
                    compressed = "1"
                elif abs(value + 1.0) < 1e-10:
                    compressed = "-1"
                else:
                    # Mantenir decimals només si són necessaris
                    formatted = f"{value:.{self.config['precision_digits']}f}"
                    # Eliminar zeros finals
                    if '.' in formatted:
                        formatted = formatted.rstrip('0').rstrip('.')
                    compressed = formatted
                
                compressed_parts.append(compressed)
            except:
                compressed_parts.append(part.strip())
        
        return ", ".join(compressed_parts)
    
    def _compress_translation_string(self, translate_string: str) -> str:
        """Comprimeix una cadena de translació."""
        
        parts = translate_string.replace('px', '').split(',')
        if len(parts) != 3:
            return translate_string
        
        compressed_parts = []
        for part in parts:
            try:
                value = float(part.strip())
                # Arrodonir a píxels sencers si és pròxim
                if abs(value - round(value)) < 0.01:
                    compressed = f"{int(round(value))}px"
                else:
                    compressed = f"{value:.1f}px"
                compressed_parts.append(compressed)
            except:
                compressed_parts.append(part.strip() + "px")
        
        return ", ".join(compressed_parts)
    
    def _compress_rotation_string(self, rotate_string: str) -> str:
        """Comprimeix una cadena de rotació."""
        
        parts = rotate_string.replace('deg', '').split(',')
        if len(parts) != 4:
            return rotate_string
        
        compressed_parts = []
        for i, part in enumerate(parts):
            try:
                value = float(part.strip())
                
                if i < 3:  # Components d'eix
                    if abs(value) < 0.01:
                        compressed = "0"
                    else:
                        compressed = f"{value:.2f}"
                else:  # Angle
                    # Arrodonir a graus sencers si és pròxim
                    if abs(value - round(value)) < 0.1:
                        compressed = f"{int(round(value))}deg"
                    else:
                        compressed = f"{value:.1f}deg"
                
                compressed_parts.append(compressed)
            except:
                compressed_parts.append(part.strip() + ("deg" if i == 3 else ""))
        
        return ", ".join(compressed_parts)
    
    def _compress_scale_string(self, scale_string: str) -> str:
        """Comprimeix una cadena d'escalat."""
        
        parts = scale_string.split(',')
        if len(parts) != 3:
            return scale_string
        
        compressed_parts = []
        for part in parts:
            try:
                value = float(part.strip())
                
                if abs(value - 1.0) < 0.001:
                    compressed = "1"
                else:
                    compressed = f"{value:.2f}"
                
                compressed_parts.append(compressed)
            except:
                compressed_parts.append(part.strip())
        
        return ", ".join(compressed_parts)
    
    def get_transformer_report(self) -> Dict[str, Any]:
        """Genera informe del transformador."""
        
        return {
            "transformer_statistics": self.stats,
            "cache_performance": {
                "cache_size": len(self.transform_cache),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate": self.stats["cache_efficiency"]
            },
            "configuration": self.config,
            "common_matrices_available": list(self.common_matrices.keys())
        }