import numpy as np
import pybullet as p


class CameraController:
    """
    Keyboard camera controller for the PyBullet debug visualizer.

    Controls (letters only; avoids PyBullet special keys like arrows):
      - Pan (ground plane, relative to camera yaw):
          W = forward, S = backward, A = left, D = right
      - Rotate:
          J = yaw left, L = yaw right
          I = pitch up, K = pitch down
      - Zoom:
          U = zoom in, O = zoom out
      - Raise/lower target:
          R = up, F = down
      - Utilities:
          SHIFT = speed boost
          P = print current camera parameters

    Notes:
      - Works only in GUI mode (p.GUI).
      - Call update() once per simulation step.
    """

    def __init__(
        self,
        dist: float = 2.5,
        yaw_deg: float = 45.0,
        pitch_deg: float = -30.0,
        target=(0.0, 0.0, 0.3),
        pan_speed: float = 0.06,      # meters per tick
        rot_speed: float = 2.0,       # degrees per tick
        zoom_speed: float = 0.06,     # meters per tick
        z_speed: float = 0.04,        # meters per tick
        boost_mult: float = 3.0,      # when shift is held
        min_dist: float = 0.2,
        max_dist: float = 50.0,
    ):
        self.dist = float(dist)
        self.yaw = float(yaw_deg)
        self.pitch = float(pitch_deg)
        self.target = np.array([float(target[0]), float(target[1]), float(target[2])], dtype=float)

        self.pan_speed = float(pan_speed)
        self.rot_speed = float(rot_speed)
        self.zoom_speed = float(zoom_speed)
        self.z_speed = float(z_speed)

        self.boost_mult = float(boost_mult)
        self.min_dist = float(min_dist)
        self.max_dist = float(max_dist)

        self._prev_p_pressed = False  # for edge-triggered printing

    def _is_down(self, events, key: str) -> bool:
        k = ord(key)
        return (k in events) and (events[k] & p.KEY_IS_DOWN)

    def _is_pressed(self, events, key: str) -> bool:
        # edge-triggered "went down"
        k = ord(key)
        return (k in events) and (events[k] & p.KEY_WAS_TRIGGERED)

    def update(self):
        events = p.getKeyboardEvents()

        
        mult = 1.0

        # --- Rotation (J/L yaw, I/K pitch) ---
        if self._is_down(events, 'j'):
            self.yaw -= self.rot_speed * mult
        if self._is_down(events, 'l'):
            self.yaw += self.rot_speed * mult

        if self._is_down(events, 'i'):
            self.pitch = min(89.0, self.pitch + self.rot_speed * mult)
        if self._is_down(events, 'k'):
            self.pitch = max(-89.0, self.pitch - self.rot_speed * mult)

        # --- Zoom (U in, O out) ---
        if self._is_down(events, 'u'):
            self.dist = max(self.min_dist, self.dist - self.zoom_speed * mult)
        if self._is_down(events, 'o'):
            self.dist = min(self.max_dist, self.dist + self.zoom_speed * mult)

        # --- Raise/lower target (R/F) ---
        if self._is_down(events, 'r'):
            self.target[2] += self.z_speed * mult
        if self._is_down(events, 'f'):
            self.target[2] = max(0.0, self.target[2] - self.z_speed * mult)

        # --- Pan target in XY (WASD) relative to camera yaw ---
        # forward direction projected onto XY plane
        yaw_rad = np.deg2rad(self.yaw)
        forward = np.array([np.cos(yaw_rad), np.sin(yaw_rad)], dtype=float)
        left = np.array([-forward[1], forward[0]], dtype=float)

        pan = self.pan_speed * mult
        if self._is_down(events, 'w'):
            self.target[:2] += pan * forward
        if self._is_down(events, 's'):
            self.target[:2] -= pan * forward
        if self._is_down(events, 'a'):
            self.target[:2] += pan * left
        if self._is_down(events, 'd'):
            self.target[:2] -= pan * left

        # --- Utility: print camera params (P) ---
        if self._is_pressed(events, 'p'):
            print(
                f"Camera: dist={self.dist:.3f}, yaw={self.yaw:.1f}, pitch={self.pitch:.1f}, "
                f"target=[{self.target[0]:.3f}, {self.target[1]:.3f}, {self.target[2]:.3f}]"
            )

        # Apply camera
        p.resetDebugVisualizerCamera(
            cameraDistance=float(self.dist),
            cameraYaw=float(self.yaw),
            cameraPitch=float(self.pitch),
            cameraTargetPosition=self.target.tolist(),
        )
