"""
Linux game-window lookup — replacement for the Windows `pygetwindow` path.

Proton/Steam games render through XWayland, so their top-level window lives on
the X11 display and can be located by name or PID with `xdotool`. We return the
same `(left, top, right, bottom)` bounding box the dxcam capture path expects.

Resolution order:
  1. An explicit --region "L,T,W,H" (most reliable; bypasses lookup entirely).
  2. xdotool search by window name (default "Cyberpunk").
  3. python-Xlib enumeration as a fallback if xdotool is unavailable.
"""
import shutil
import subprocess


def region_from_str(region_str):
    """Parse 'left,top,width,height' into (left, top, right, bottom)."""
    parts = [int(x.strip()) for x in region_str.split(",")]
    if len(parts) != 4:
        raise ValueError("--region must be 'left,top,width,height'")
    left, top, width, height = parts
    return (left, top, left + width, top + height)


def _xdotool_geometry(win_id):
    out = subprocess.check_output(
        ["xdotool", "getwindowgeometry", "--shell", str(win_id)],
        text=True,
    )
    vals = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip()
    x, y = int(vals["X"]), int(vals["Y"])
    w, h = int(vals["WIDTH"]), int(vals["HEIGHT"])
    return (x, y, x + w, y + h)


def _find_with_xdotool(name):
    # Search visible top-level windows whose name matches (case-insensitive).
    try:
        out = subprocess.check_output(
            ["xdotool", "search", "--onlyvisible", "--name", name],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        out = ""
    ids = [int(x) for x in out.split()]
    if not ids:
        return None
    # Pick the largest matching window (the game viewport, not a helper window).
    best, best_area = None, -1
    for wid in ids:
        try:
            l, t, r, b = _xdotool_geometry(wid)
        except Exception:
            continue
        area = (r - l) * (b - t)
        if area > best_area:
            best, best_area = (l, t, r, b), area
    return best


def _find_with_xlib(name):
    try:
        from Xlib import display, X  # noqa: F401
    except Exception:
        return None
    d = display.Display()
    root = d.screen().root
    name_l = name.lower()
    best, best_area = None, -1

    def walk(window):
        nonlocal best, best_area
        try:
            wm_name = window.get_wm_name()
        except Exception:
            wm_name = None
        if wm_name and name_l in wm_name.lower():
            geom = window.get_geometry()
            # Translate to absolute root coordinates.
            coords = window.translate_coords(root, 0, 0)
            x = -coords.x
            y = -coords.y
            area = geom.width * geom.height
            if area > best_area:
                best, best_area = (x, y, x + geom.width, y + geom.height), area
        try:
            for child in window.query_tree().children:
                walk(child)
        except Exception:
            pass

    walk(root)
    return best


def find_game_window(window_name="Cyberpunk"):
    """Return (left, top, right, bottom) for the game window, or raise."""
    if shutil.which("xdotool"):
        box = _find_with_xdotool(window_name)
        if box:
            return box
    box = _find_with_xlib(window_name)
    if box:
        return box
    raise RuntimeError(
        f"Could not find a window matching '{window_name}'. "
        f"Pass --region 'left,top,width,height' to capture explicitly, or make "
        f"sure the game is running in a desktop (XWayland) window."
    )
