import numpy as np


class UpperBodyRetarget:
    """
    Map head motion/orientation to waist yaw + pitch.

    - Forward (팔 뻗는 쪽): measured along robot Z in this setup.
      Use baseline z0 and forward_delta = (z0 - z_now) so moving forward
      (z_now smaller) yields positive delta.
    - Downward: measured along robot Y (height baseline y0). Use down_delta =
      max(0, y0 - y_now) so only downward contributes.
    - Yaw: extracted from head rotation around robot Z axis, relative to
      initial head orientation.
    - Waist roll fixed to 0 for stability.
    """

    def __init__(self, height: float = 1.60, waist_height_ratio: float = 0.530,
                 pitch_gain_forward: float = 6.0, pitch_gain_down: float = 6.0,
                 yaw_gain: float = 1.0,
                 clip: float = np.pi / 6,
                 deadzone_forward: float = 0.0, deadzone_down: float = 0.0,
                 deadzone_yaw: float = 0.0):
        self.height = float(height)
        self.waist_height_ratio = float(waist_height_ratio)
        self.waist_height = self.height * self.waist_height_ratio
        self.pitch_gain_forward = float(pitch_gain_forward)
        self.pitch_gain_down = float(pitch_gain_down)
        self.yaw_gain = float(yaw_gain)
        self.clip = float(clip)
        self.deadzone_forward = float(deadzone_forward)
        self.deadzone_down = float(deadzone_down)
        self.deadzone_yaw = float(deadzone_yaw)
        self._z0 = None  # forward baseline
        self._y0 = None  # height baseline
        self._yaw0 = None  # head yaw baseline (rad)

    @staticmethod
    def _extract_yaw_from_rot(mat: np.ndarray) -> float:
        """
        Extract yaw (rotation around robot Z) from a 4x4 SE(3) pose matrix.

        Uses standard ZYX convention: R = Rz(yaw) * Ry(pitch) * Rx(roll).
        """
        R = mat[:3, :3]
        # Guard against invalid matrices
        if not np.isfinite(R).all():
            return 0.0
        # atan2 handles normalization internally
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def solve_upper_body_angles(self, head: np.ndarray):
        """
        Return [waist_yaw, waist_pitch, waist_roll].

        - waist_pitch = k_fwd * (z0 - z_now) + k_down * max(0, y0 - y_now)
        - waist_yaw   = yaw_gain * (head_yaw - head_yaw_0)
        """
        if head is None or head.shape != (4, 4):
            return np.zeros(3)

        z_now = float(head[2, 3])  # forward/back (this setup)
        y_now = float(head[1, 3])  # height (baseline at start)

        if self._z0 is None:
            self._z0 = z_now
        if self._y0 is None:
            self._y0 = y_now
        # Baseline head yaw from first valid frame
        yaw_now = self._extract_yaw_from_rot(head)
        if self._yaw0 is None:
            self._yaw0 = yaw_now

        forward_delta = (self._z0 - z_now)
        down_delta = max(0.0, self._y0 - y_now)

        if abs(forward_delta) < self.deadzone_forward:
            forward_delta = 0.0
        if abs(down_delta) < self.deadzone_down:
            down_delta = 0.0

        # Yaw delta (wrap to [-pi, pi] to avoid drift)
        yaw_delta = yaw_now - self._yaw0
        yaw_delta = (yaw_delta + np.pi) % (2 * np.pi) - np.pi
        if abs(yaw_delta) < self.deadzone_yaw:
            yaw_delta = 0.0

        waist_pitch = self.pitch_gain_forward * forward_delta + \
                      self.pitch_gain_down * down_delta
        waist_yaw = self.yaw_gain * yaw_delta

        q_upper_body = np.array([waist_yaw, waist_pitch, 0.0], dtype=float)
        q_upper_body = np.clip(q_upper_body, -self.clip, self.clip)
        return q_upper_body
