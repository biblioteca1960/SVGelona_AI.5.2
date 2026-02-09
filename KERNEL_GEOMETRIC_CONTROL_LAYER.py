# ==========================================================
# SVGelona_AI 5.2
# GEOMETRIC CONTROL LAYER
# Objective: Angle-based global kernel coherence
# ==========================================================

import numpy as np

class KernelGeometricController:

    def __init__(self):
        self.reference_frame = np.identity(3)
        self.angular_threshold = 1e-5

    def enforce_global_alignment(self, state_vectors):
        """
        Ensures that all kernel subsystems remain
        geometrically aligned within tolerance.
        """
        corrected_vectors = []

        for v in state_vectors:
            angle = self._angle_to_reference(v)

            if abs(angle) > self.angular_threshold:
                v = self._apply_rotation_correction(v, angle)

            corrected_vectors.append(v)

        return corrected_vectors

    def _angle_to_reference(self, vector):
        ref = self.reference_frame[:, 0]
        dot = np.dot(vector, ref)
        norm = np.linalg.norm(vector) * np.linalg.norm(ref)

        if norm == 0:
            return 0.0

        return np.arccos(np.clip(dot / norm, -1.0, 1.0))

    def _apply_rotation_correction(self, vector, angle):
        """
        Minimal corrective rotation around Z-axis.
        """
        rotation_matrix = np.array([
            [np.cos(-angle), -np.sin(-angle), 0],
            [np.sin(-angle),  np.cos(-angle), 0],
            [0,               0,              1]
        ])

        return rotation_matrix @ vector
