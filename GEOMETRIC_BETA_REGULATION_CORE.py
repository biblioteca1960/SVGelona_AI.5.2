# ==========================================================
# SVGelona_AI 5.2
# GEOMETRIC BETA REGULATION CORE
# Objective: Angle-based stabilization of beta-driven entropy
# ==========================================================

import numpy as np

class GeometricBetaRegulationCore:

    def __init__(self):
        # Critical parameters
        self.beta_target = 0.5
        self.angle_tolerance = 1e-6

        # Inverse golden damping (numerical stabilizer, non-symbolic)
        self.damping_factor = 0.61803398875

        # Phase memory for loop detection
        self.phase_memory = []
        self.memory_window = 128

    def compute_projection_angle(self, vector_a, vector_b):
        """
        Computes the angle between two vectors in radians.
        Used to detect geometric misalignment in beta evolution.
        """
        dot = np.dot(vector_a, vector_b)
        norm = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)

        if norm == 0:
            return 0.0

        cos_theta = np.clip(dot / norm, -1.0, 1.0)
        return np.arccos(cos_theta)

    def stabilize_beta(self, alpha_output, environmental_entropy):
        """
        Core beta regulation function.
        Beta is treated as a geometric projection problem.
        """
        beta_raw = alpha_output + environmental_entropy

        # Angular correction model
        reference_axis = np.array([1.0, 0.0, 0.0])
        beta_vector = np.array([beta_raw, 1.0 - beta_raw, 0.0])

        angle = self.compute_projection_angle(beta_vector, reference_axis)

        # Apply damping if misalignment exceeds tolerance
        if abs(angle) > self.angle_tolerance:
            beta_corrected = beta_raw * self.damping_factor
        else:
            beta_corrected = beta_raw

        # Store phase for loop detection
        self._register_phase(beta_corrected)

        return beta_corrected

    def _register_phase(self, value):
        self.phase_memory.append(value)
        if len(self.phase_memory) > self.memory_window:
            self.phase_memory.pop(0)

    def detect_phase_loop(self):
        """
        Detects numerical stagnation or oscillatory loops.
        """
        if len(self.phase_memory) < 10:
            return False

        recent = self.phase_memory[-1]
        repetitions = sum(
            1 for v in self.phase_memory[:-1]
            if abs(v - recent) < self.angle_tolerance
        )

        return repetitions > 5
