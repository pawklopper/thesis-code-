import ctypes as ct
import threading
import time
import numpy as np


#
import time
import numpy as np




class Sigma7ImpedanceController:
    """
    3D Translation impedance controller centered at (0,0,0).

    - Reads full 3D position and linear velocity.
    - Computes 3D displacement from (0,0,0).
    - Applies 3D PD force to pull device back toward (0,0,0).
    - Exposes ONLY XY displacement and XY force via get_haptic_data() for a 2D puck simulation.
    """

    def __init__(
        self,
        lib_path: str,
        Kp=100.0,
        Dp=20.0,
        MAX_FORCE=40.0,            # kept for compatibility with your outer code
        MAX_DEVICE_FORCE=15.0,     # device force magnitude limit (N)
        frequency=4000,
        libdrd_path: str = None,
        home_on_start: bool = True,
        recenter_on_start: bool = True,
        recenter_force: float = 4.0,
        debug_print_hz: float = 10.0,
    ):
        self._dhd = ct.CDLL(lib_path)

        self.Kp = float(Kp)
        self.Dp = float(Dp)
        self.MAX_FORCE = float(MAX_FORCE)
        self.MAX_DEVICE_FORCE = float(MAX_DEVICE_FORCE)
        self.frequency = float(frequency)

        # Explicit device ID
        self._id = ct.c_byte(0)

        # 3D center and state
        self._center_xyz = np.array([0.0, 0.0, 0.0], dtype=float)
        self._displacement_xyz = np.zeros(3, dtype=float)
        self._force_xyz = np.zeros(3, dtype=float)

        self._running = False
        self._lock = threading.Lock()
        self._thread = None

        # Optional fields kept (not used here)
        self._libdrd_path = libdrd_path
        self._home_on_start = bool(home_on_start)

        # One-time startup recenter settings
        self._recenter_on_start = bool(recenter_on_start)
        self._recenter_force = float(recenter_force)

        # Debug printing throttle
        self._debug_print_hz = float(debug_print_hz)

        self._bind()

    def _bind(self):
        self._dhd.dhdGetDeviceCount.restype = ct.c_int

        self._dhd.dhdOpenID.argtypes = [ct.c_int]
        self._dhd.dhdOpenID.restype = ct.c_int

        self._dhd.dhdClose.argtypes = [ct.c_byte]
        self._dhd.dhdClose.restype = ct.c_int

        self._dhd.dhdEnableForce.argtypes = [ct.c_int, ct.c_byte]
        self._dhd.dhdEnableForce.restype = ct.c_int

        self._dhd.dhdGetPosition.argtypes = [
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_byte
        ]
        self._dhd.dhdGetPosition.restype = ct.c_int

        self._dhd.dhdGetLinearVelocity.argtypes = [
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_byte
        ]
        self._dhd.dhdGetLinearVelocity.restype = ct.c_int

        self._dhd.dhdSetForce.argtypes = [ct.c_double, ct.c_double, ct.c_double, ct.c_byte]
        self._dhd.dhdSetForce.restype = ct.c_int

        self._dhd.dhdErrorGetLastStr.restype = ct.c_char_p

    def start(self):
        """Starts the haptic thread. Device open + force loop happen inside that thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._haptic_loop, daemon=True)
        self._thread.start()

    def close(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def get_haptic_data(self):
        """Return ONLY XY displacement/force for the puck simulation."""
        with self._lock:
            return self._displacement_xyz[:2].copy(), self._force_xyz[:2].copy()

    def get_haptic_data_3d(self):
        """Optional: full 3D displacement/force if you ever need it."""
        with self._lock:
            return self._displacement_xyz.copy(), self._force_xyz.copy()

    def _err(self) -> str:
        msg = self._dhd.dhdErrorGetLastStr()
        return msg.decode() if msg else "unknown error"

    @staticmethod
    def _clip_vec(v: np.ndarray, max_norm: float) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n > max_norm and n > 1e-12:
            return (v / n) * max_norm
        return v

    def _run_soft_recenter(
        self,
        target_xyz=np.array([0.0, 0.0, 0.0]),
        Kp_recenter=40.0,
        Dp_recenter=8.0,
        max_recenter_force=4.0,
        timeout=5.0,
    ):
        """
        Softly pulls the handle toward target_xyz with capped force.
        Translation only.
        """
        px, py, pz = ct.c_double(), ct.c_double(), ct.c_double()
        vx, vy, vz = ct.c_double(), ct.c_double(), ct.c_double()

        t0 = time.time()
        target_xyz = np.array(target_xyz, dtype=float)

        while self._running and (time.time() - t0) < timeout:
            ok_pos = self._dhd.dhdGetPosition(ct.byref(px), ct.byref(py), ct.byref(pz), self._id) >= 0
            ok_vel = self._dhd.dhdGetLinearVelocity(ct.byref(vx), ct.byref(vy), ct.byref(vz), self._id) >= 0
            if not (ok_pos and ok_vel):
                time.sleep(1.0 / max(self.frequency, 1.0))
                continue

            curr = np.array([px.value, py.value, pz.value], dtype=float)
            vel = np.array([vx.value, vy.value, vz.value], dtype=float)

            err = curr - target_xyz
            f = -Kp_recenter * err - Dp_recenter * vel
            f = self._clip_vec(f, max_recenter_force)

            self._dhd.dhdSetForce(float(f[0]), float(f[1]), float(f[2]), self._id)
            time.sleep(1.0 / max(self.frequency, 1.0))

    def _haptic_loop(self):
        if self._dhd.dhdGetDeviceCount() <= 0:
            print("No haptic device detected")
            self._running = False
            return

        if self._dhd.dhdOpenID(int(self._id.value)) < 0:
            print("dhdOpenID failed:", self._err())
            self._running = False
            return

        if self._dhd.dhdEnableForce(1, self._id) < 0:
            print("dhdEnableForce failed:", self._err())
            self._dhd.dhdClose(self._id)
            self._running = False
            return

        print("✅ Translation PD centered at (0,0,0)")

        if self._recenter_on_start:
            self._recenter_on_start = False
            print(f"Starting gentle recenter to (0,0,0) with cap {self._recenter_force} N")
            self._run_soft_recenter(
                target_xyz=np.array([0.0, 0.0, 0.0], dtype=float),
                max_recenter_force=self._recenter_force
            )

        px, py, pz = ct.c_double(), ct.c_double(), ct.c_double()
        vx, vy, vz = ct.c_double(), ct.c_double(), ct.c_double()

        target_period = 1.0 / max(self.frequency, 1.0)

        debug_period = (1.0 / self._debug_print_hz) if self._debug_print_hz > 0 else None
        debug_period = None
        last_debug_t = time.time()


        target_period = 1.0 / max(self.frequency, 1.0)

        # timing stats
        t_prev = time.perf_counter()
        t_window_start = t_prev
        iters = 0

        dt_min = float("inf")
        dt_max = 0.0
        dt_sum = 0.0
        dt2_sum = 0.0  # for RMS/jitter estimate

        report_every_s = 1.0


        try:
            while self._running:
                ok_pos = self._dhd.dhdGetPosition(ct.byref(px), ct.byref(py), ct.byref(pz), self._id) >= 0
                ok_vel = self._dhd.dhdGetLinearVelocity(ct.byref(vx), ct.byref(vy), ct.byref(vz), self._id) >= 0
                if not (ok_pos and ok_vel):
                    time.sleep(target_period)
                    continue

                t_now = time.perf_counter()
                dt = t_now - t_prev
                t_prev = t_now

                iters += 1
                dt_min = min(dt_min, dt)
                dt_max = max(dt_max, dt)
                dt_sum += dt
                dt2_sum += dt * dt

                # Report once per second
                if (t_now - t_window_start) >= report_every_s:
                    elapsed = t_now - t_window_start
                    hz = iters / elapsed
                    dt_mean = dt_sum / iters
                    dt_rms = (dt2_sum / iters) ** 0.5
                    # "jitter" approximation: RMS deviation from mean
                    jitter_rms = (max(dt_rms * dt_rms - dt_mean * dt_mean, 0.0)) ** 0.5

                    print(
                        f"Achieved: {hz:7.1f} Hz | "
                        f"dt mean: {dt_mean*1e6:7.1f} us | "
                        f"dt min/max: {dt_min*1e6:7.1f}/{dt_max*1e6:7.1f} us | "
                        f"jitter(rms): {jitter_rms*1e6:7.1f} us"
                    )

                    # reset window
                    t_window_start = t_now
                    iters = 0
                    dt_min = float("inf")
                    dt_max = 0.0
                    dt_sum = 0.0
                    dt2_sum = 0.0


                now = time.time()

                curr_pos = np.array([px.value, py.value, pz.value], dtype=float)
                vel_xyz = np.array([vx.value, vy.value, vz.value], dtype=float)

                displacement_xyz = curr_pos - self._center_xyz
                f_calc = -self.Kp * displacement_xyz - self.Dp * vel_xyz
                f_cmd = self._clip_vec(f_calc, self.MAX_DEVICE_FORCE)

                with self._lock:
                    self._displacement_xyz = displacement_xyz
                    self._force_xyz = f_cmd

                rc = self._dhd.dhdSetForce(float(f_cmd[0]), float(f_cmd[1]), float(f_cmd[2]), self._id)
                if rc < 0:
                    print("dhdSetForce failed:", self._err())
                    self._running = False
                    break

                if debug_period is not None and (now - last_debug_t) >= debug_period:
                    last_debug_t = now
                    print(f"disp_xyz: {displacement_xyz} | f_cmd: {f_cmd}")

                time.sleep(target_period)

        finally:
            self._dhd.dhdSetForce(0.0, 0.0, 0.0, self._id)
            self._dhd.dhdClose(self._id)
