"""
Linux screen capture — drop-in replacement for `dxcam`.

`dxcam` uses the Windows DirectX Desktop Duplication API and does not exist on
Linux. This module mirrors the tiny slice of the dxcam API the play scripts use
(`create(...)`, `.start()`, `.get_latest_frame()`, `.stop()`) on top of `mss`,
which grabs from the X11 server. Under a Wayland session, Proton/Steam games run
as XWayland clients, so their window is present on the X11 display and can be
grabbed by region — no portal handshake required.

    camera = create(output_color="RGB", region=(left, top, right, bottom))
    camera.start(target_fps=60, video_mode=True)   # no-op flags, kept for parity
    frame = camera.get_latest_frame()              # HxWx3 RGB uint8 ndarray
    camera.stop()

`mss` is not thread-safe across threads, so the grabber is created lazily in the
thread that first calls `get_latest_frame()` (the play loop's main thread).
"""
import threading

import numpy as np


class _MssCamera:
    def __init__(self, region=None, output_color="RGB"):
        # dxcam region is (left, top, right, bottom); mss wants a dict.
        self._region = region
        self._output_color = output_color.upper()
        self._local = threading.local()
        self._monitor = None  # resolved per-thread alongside the sct instance

    def _sct(self):
        sct = getattr(self._local, "sct", None)
        if sct is None:
            import mss  # imported lazily so `serve`-only installs don't need it
            sct = mss.mss()
            self._local.sct = sct
            if self._region is not None:
                left, top, right, bottom = self._region
                self._local.monitor = {
                    "left": int(left),
                    "top": int(top),
                    "width": int(right - left),
                    "height": int(bottom - top),
                }
            else:
                # Whole primary monitor (index 1 in mss; 0 is the union of all).
                self._local.monitor = sct.monitors[1]
        return sct

    def start(self, *args, **kwargs):
        # dxcam spins up a capture thread here; mss grabs on demand, so this only
        # warms the per-thread grabber. Flags kept for call-site compatibility.
        self._sct()

    def get_latest_frame(self):
        sct = self._sct()
        raw = sct.grab(self._local.monitor)
        # mss returns BGRA; drop alpha and order channels to match output_color.
        frame = np.asarray(raw, dtype=np.uint8)[:, :, :3]
        if self._output_color == "RGB":
            frame = frame[:, :, ::-1]  # BGR -> RGB
        return np.ascontiguousarray(frame)

    def stop(self):
        sct = getattr(self._local, "sct", None)
        if sct is not None:
            try:
                sct.close()
            except Exception:
                pass
            self._local.sct = None


def create(output_color="RGB", region=None, **_ignored):
    """dxcam.create-compatible factory. Extra dxcam kwargs are accepted/ignored."""
    return _MssCamera(region=region, output_color=output_color)
