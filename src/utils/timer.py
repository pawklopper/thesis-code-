# utils/timer_window.py
import time
import numpy as np

try:
    import cv2
except ImportError as e:
    raise ImportError(
        "OpenCV is required. Install with: pip install opencv-python"
    ) from e


def format_mmss(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60.0)
    s = int(seconds % 60.0)
    return f"{m:02d}:{s:02d}"


class EpisodeTimerWindow:
    """
    Simple separate window timer using OpenCV.
    - start_episode(): call after each env.reset()
    - update(): call each loop (throttled)
    - close(): call on shutdown
    """

    def __init__(
        self,
        window_name="Episode Timer",
        update_hz=10.0,
        width=360,
        height=120,
        font_scale=1.6,
        thickness=2,
        show_global=False,
    ):
        self.window_name = window_name
        self.update_period = 1.0 / float(update_hz)
        self.width = int(width)
        self.height = int(height)
        self.font_scale = float(font_scale)
        self.thickness = int(thickness)
        self.show_global = bool(show_global)

        self._t0_global = time.time()
        self._t0_episode = time.time()
        self._last_update = 0.0
        self._opened = False

    def _ensure_open(self):
        if not self._opened:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.width, self.height)

            # Put it e.g. on the right side of the desktop:
            cv2.moveWindow(self.window_name, -1600, 1200)  # adjust X,Y to your screen layout

            self._opened = True


    def start_episode(self):
        self._t0_episode = time.time()
        self._last_update = 0.0  # force a refresh next update()

    def update(self):
        now = time.time()
        # Keep the window responsive even when throttling
        if self._opened:
            cv2.waitKey(1)

        if (now - self._last_update) < self.update_period:
            return

        self._last_update = now
        self._ensure_open()

        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        ep = now - self._t0_episode
        ep_txt = f"EP  {format_mmss(ep)}"

        cv2.putText(
            img,
            ep_txt,
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            (255, 255, 255),
            self.thickness,
            cv2.LINE_AA,
        )

        if self.show_global:
            gl = now - self._t0_global
            gl_txt = f"ALL {format_mmss(gl)}"
            cv2.putText(
                img,
                gl_txt,
                (15, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (200, 200, 200),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(self.window_name, img)
        cv2.waitKey(1)

    def close(self):
        if self._opened:
            cv2.destroyWindow(self.window_name)
            self._opened = False
