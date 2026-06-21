"""
Wayland-native screen capture via the xdg-desktop-portal ScreenCast API + PipeWire.

On a Wayland session (KWin/Plasma here) the X11 root window has no real pixels, so
mss/X11 capture returns black for everything, including XWayland games like
Cyberpunk under Proton. The supported way to capture is the ScreenCast portal:
it authorises a source (a window, a screen, or a region) and hands back a PipeWire
node, which GStreamer's `pipewiresrc` turns into frames.

Same small interface as the mss/dxcam backends so it drops into the play loop:

    cam = create(region=None)        # region is ignored; the portal picks the source
    cam.start()                      # pops the KDE picker the first time
    frame = cam.get_latest_frame()   # HxWx3 RGB uint8 ndarray (or None until first frame)
    cam.stop()

The first run shows the portal's "select what to share" dialog. Pick the Cyberpunk
window (or the whole screen). The returned restore token is saved to
~/.config/nitrogen/screencast.token so later runs reuse the choice without asking.

Requires: PyGObject (gi), GStreamer with the pipewiresrc plugin, a running
PipeWire + xdg-desktop-portal. The game must be visible (not minimised).
"""
import os
import threading

import numpy as np

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gio, GLib, Gst, GstApp  # noqa: E402

Gst.init(None)

_PORTAL = "org.freedesktop.portal.Desktop"
_PATH = "/org/freedesktop/portal/desktop"
_SCAST = "org.freedesktop.portal.ScreenCast"
_TOKEN_FILE = os.path.expanduser("~/.config/nitrogen/screencast.token")


def _load_token():
    try:
        with open(_TOKEN_FILE) as f:
            return f.read().strip() or None
    except OSError:
        return None


def _save_token(tok):
    if not tok:
        return
    os.makedirs(os.path.dirname(_TOKEN_FILE), exist_ok=True)
    with open(_TOKEN_FILE, "w") as f:
        f.write(tok)


class PipeWireCamera:
    def __init__(self, region=None, output_color="RGB"):
        # region/output_color kept for interface parity; the portal owns the source
        # and we always deliver RGB.
        self._bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        self._sender = self._bus.get_unique_name()[1:].replace(".", "_")
        self._seq = 0
        self._session = None
        self._pipeline = None
        self._lock = threading.Lock()
        self._frame = None
        self._loop = None
        self._thread = None

    # --- portal request/response plumbing -----------------------------------
    def _token(self, prefix):
        self._seq += 1
        return f"{prefix}{self._seq}"

    def _request(self, method, fixed_args, options):
        """Call a ScreenCast method that returns a Request, wait for its Response."""
        handle_token = self._token("req")
        options["handle_token"] = GLib.Variant("s", handle_token)
        req_path = f"{_PATH}/request/{self._sender}/{handle_token}"
        loop = GLib.MainLoop()
        result = {}

        def on_response(conn, sender, path, iface, signal, params):
            code, results = params.unpack()
            result["code"] = code
            result["results"] = results
            loop.quit()

        sub = self._bus.signal_subscribe(
            _PORTAL, "org.freedesktop.portal.Request", "Response", req_path,
            None, Gio.DBusSignalFlags.NONE, on_response)

        args = list(fixed_args) + [options]
        self._bus.call_sync(_PORTAL, _PATH, _SCAST, method,
                            GLib.Variant(self._sig(method), tuple(args)),
                            None, Gio.DBusCallFlags.NONE, -1, None)
        loop.run()
        self._bus.signal_unsubscribe(sub)
        if result.get("code") != 0:
            raise RuntimeError(f"portal {method} failed/cancelled (code={result.get('code')})")
        return result["results"]

    @staticmethod
    def _sig(method):
        return {
            "CreateSession": "(a{sv})",
            "SelectSources": "(oa{sv})",
            "Start": "(osa{sv})",
        }[method]

    # --- lifecycle ----------------------------------------------------------
    def start(self, target_fps=60, video_mode=True):
        res = self._request("CreateSession", [], {
            "session_handle_token": GLib.Variant("s", self._token("sess")),
        })
        self._session = res["session_handle"]

        select = {
            "types": GLib.Variant("u", 1 | 2),       # 1=monitor, 2=window
            "multiple": GLib.Variant("b", False),
            "cursor_mode": GLib.Variant("u", 2),     # embed the cursor in the frames
            "persist_mode": GLib.Variant("u", 2),    # persist until revoked
        }
        tok = _load_token()
        if tok:
            select["restore_token"] = GLib.Variant("s", tok)
        self._request("SelectSources", [self._session], select)

        started = self._request("Start", [self._session, ""], {})
        _save_token(started.get("restore_token"))
        streams = started["streams"]
        if not streams:
            raise RuntimeError("portal returned no streams (nothing was shared)")
        node_id = streams[0][0]

        fd = self._open_pipewire_remote()
        self._build_pipeline(fd, node_id)

    def _open_pipewire_remote(self):
        ret, fdlist = self._bus.call_with_unix_fd_list_sync(
            _PORTAL, _PATH, _SCAST, "OpenPipeWireRemote",
            GLib.Variant("(oa{sv})", (self._session, {})),
            GLib.VariantType.new("(h)"), Gio.DBusCallFlags.NONE, -1, None, None)
        fd_index = ret.unpack()[0]
        return fdlist.get(fd_index)

    def _build_pipeline(self, fd, node_id):
        desc = (f"pipewiresrc fd={fd} path={node_id} keepalive-time=1000 ! "
                f"videoconvert ! video/x-raw,format=RGB ! "
                f"appsink name=sink emit-signals=true max-buffers=2 drop=true sync=false")
        self._pipeline = Gst.parse_launch(desc)
        sink = self._pipeline.get_by_name("sink")
        sink.connect("new-sample", self._on_sample)
        self._pipeline.set_state(Gst.State.PLAYING)
        self._loop = GLib.MainLoop()
        self._thread = threading.Thread(target=self._loop.run, daemon=True)
        self._thread.start()

    def _on_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        w = caps.get_value("width")
        h = caps.get_value("height")
        ok, info = buf.map(Gst.MapFlags.READ)
        if ok:
            # RGB rows may be padded to a 4-byte stride; trim to w*3 per row.
            rowstride = (w * 3 + 3) & ~3
            data = np.frombuffer(info.data, dtype=np.uint8)
            try:
                arr = data[:h * rowstride].reshape(h, rowstride)[:, : w * 3].reshape(h, w, 3)
                with self._lock:
                    self._frame = arr.copy()
            except ValueError:
                pass
            buf.unmap(info)
        return Gst.FlowReturn.OK

    def get_latest_frame(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
        if self._loop is not None:
            self._loop.quit()
        if self._session is not None:
            try:
                self._bus.call_sync(_PORTAL, self._session, "org.freedesktop.portal.Session",
                                    "Close", None, None, Gio.DBusCallFlags.NONE, -1, None)
            except GLib.Error:
                pass


def create(output_color="RGB", region=None, **_ignored):
    """dxcam.create-compatible factory for the PipeWire/portal backend."""
    return PipeWireCamera(region=region, output_color=output_color)
