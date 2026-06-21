"""
Linux virtual Xbox 360 gamepad — drop-in replacement for `vgamepad`.

On Windows the project uses `vgamepad` (backed by the ViGEmBus kernel driver) to
present a virtual Xbox 360 controller. There is no ViGEmBus on Linux, but the
Linux kernel exposes the same capability through `uinput`: we create a virtual
input device that advertises the Microsoft X-Box 360 pad vendor/product and the
standard button/axis layout. Steam Input (and any Proton game) then sees a real
Xbox 360 controller.

The public surface mirrors the subset of `vgamepad` that the play scripts use, so
`import nitrogen.linux_gamepad as vg` is a drop-in for `import vgamepad as vg`:

    gamepad = vg.VX360Gamepad()
    gamepad.left_joystick(x_value=..., y_value=...)   # int16, [-32768, 32767]
    gamepad.right_joystick(x_value=..., y_value=...)
    gamepad.left_trigger(value=...)                   # uint8, [0, 255]
    gamepad.right_trigger(value=...)
    gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    gamepad.update()                                  # flush a frame
    gamepad.reset()                                   # neutral state

Requires: python-evdev, and a writable /dev/uinput (on Bazzite the desktop user
has direct access; otherwise add a udev rule / the `input` group).
"""
from evdev import UInput, ecodes as e, AbsInfo


# --- vgamepad-compatible button namespace -----------------------------------
# Names match vgamepad's XUSB_BUTTON; values are the Linux evdev key codes that
# the real xpad driver reports, so Steam applies its native Xbox 360 mapping.
# Face buttons are POSITION based (SOUTH=A, EAST=B, WEST=X, NORTH=Y), which is
# how NitroGen's token set names them.
class _XUSB_BUTTON:
    XUSB_GAMEPAD_A = e.BTN_SOUTH              # bottom  (A)
    XUSB_GAMEPAD_B = e.BTN_EAST               # right   (B)
    XUSB_GAMEPAD_X = e.BTN_NORTH              # left    (X)  -> BTN_X == BTN_NORTH
    XUSB_GAMEPAD_Y = e.BTN_WEST               # top     (Y)  -> BTN_Y == BTN_WEST
    XUSB_GAMEPAD_LEFT_SHOULDER = e.BTN_TL
    XUSB_GAMEPAD_RIGHT_SHOULDER = e.BTN_TR
    XUSB_GAMEPAD_BACK = e.BTN_SELECT
    XUSB_GAMEPAD_START = e.BTN_START
    XUSB_GAMEPAD_GUIDE = e.BTN_MODE
    XUSB_GAMEPAD_LEFT_THUMB = e.BTN_THUMBL
    XUSB_GAMEPAD_RIGHT_THUMB = e.BTN_THUMBR
    # D-pad is delivered as a hat axis on a real 360 pad; handled specially.
    XUSB_GAMEPAD_DPAD_UP = "DPAD_UP"
    XUSB_GAMEPAD_DPAD_DOWN = "DPAD_DOWN"
    XUSB_GAMEPAD_DPAD_LEFT = "DPAD_LEFT"
    XUSB_GAMEPAD_DPAD_RIGHT = "DPAD_RIGHT"


XUSB_BUTTON = _XUSB_BUTTON

_KEY_BUTTONS = [
    e.BTN_SOUTH, e.BTN_EAST, e.BTN_NORTH, e.BTN_WEST,
    e.BTN_TL, e.BTN_TR, e.BTN_SELECT, e.BTN_START, e.BTN_MODE,
    e.BTN_THUMBL, e.BTN_THUMBR,
]

_ABS_AXES = [
    (e.ABS_X, AbsInfo(value=0, min=-32768, max=32767, fuzz=16, flat=128, resolution=0)),
    (e.ABS_Y, AbsInfo(value=0, min=-32768, max=32767, fuzz=16, flat=128, resolution=0)),
    (e.ABS_RX, AbsInfo(value=0, min=-32768, max=32767, fuzz=16, flat=128, resolution=0)),
    (e.ABS_RY, AbsInfo(value=0, min=-32768, max=32767, fuzz=16, flat=128, resolution=0)),
    (e.ABS_Z, AbsInfo(value=0, min=0, max=255, fuzz=0, flat=0, resolution=0)),
    (e.ABS_RZ, AbsInfo(value=0, min=0, max=255, fuzz=0, flat=0, resolution=0)),
    (e.ABS_HAT0X, AbsInfo(value=0, min=-1, max=1, fuzz=0, flat=0, resolution=0)),
    (e.ABS_HAT0Y, AbsInfo(value=0, min=-1, max=1, fuzz=0, flat=0, resolution=0)),
]


class VX360Gamepad:
    """Virtual Xbox 360 controller exposing the vgamepad API subset used here."""

    def __init__(self):
        cap = {
            e.EV_KEY: _KEY_BUTTONS,
            e.EV_ABS: _ABS_AXES,
        }
        # vendor/product/version match a wired Xbox 360 pad so SDL/Steam map it.
        self._ui = UInput(
            cap,
            name="Microsoft X-Box 360 pad",
            vendor=0x045E,
            product=0x028E,
            version=0x0114,
        )
        self._pending_keys = set()   # evdev key codes to hold this frame
        self._pending_dpad = set()   # "DPAD_*" tokens
        self._lx = self._ly = self._rx = self._ry = 0
        self._lt = self._rt = 0
        self.update()

    # --- vgamepad-compatible mutators (do not emit until update()) -----------
    def left_joystick(self, x_value, y_value):
        self._lx, self._ly = int(x_value), int(y_value)

    def right_joystick(self, x_value, y_value):
        self._rx, self._ry = int(x_value), int(y_value)

    def left_trigger(self, value):
        self._lt = max(0, min(255, int(value)))

    def right_trigger(self, value):
        self._rt = max(0, min(255, int(value)))

    def press_button(self, button):
        if isinstance(button, str) and button.startswith("DPAD_"):
            self._pending_dpad.add(button)
        else:
            self._pending_keys.add(button)

    def reset(self):
        self._pending_keys.clear()
        self._pending_dpad.clear()
        self._lx = self._ly = self._rx = self._ry = 0
        self._lt = self._rt = 0

    # --- flush one controller frame -----------------------------------------
    def update(self):
        ui = self._ui
        ui.write(e.EV_ABS, e.ABS_X, _clamp16(self._lx))
        ui.write(e.EV_ABS, e.ABS_Y, _clamp16(self._ly))
        ui.write(e.EV_ABS, e.ABS_RX, _clamp16(self._rx))
        ui.write(e.EV_ABS, e.ABS_RY, _clamp16(self._ry))
        ui.write(e.EV_ABS, e.ABS_Z, self._lt)
        ui.write(e.EV_ABS, e.ABS_RZ, self._rt)

        hat_x = (1 if "DPAD_RIGHT" in self._pending_dpad else 0) - \
                (1 if "DPAD_LEFT" in self._pending_dpad else 0)
        hat_y = (1 if "DPAD_DOWN" in self._pending_dpad else 0) - \
                (1 if "DPAD_UP" in self._pending_dpad else 0)
        ui.write(e.EV_ABS, e.ABS_HAT0X, hat_x)
        ui.write(e.EV_ABS, e.ABS_HAT0Y, hat_y)

        for code in _KEY_BUTTONS:
            ui.write(e.EV_KEY, code, 1 if code in self._pending_keys else 0)

        ui.syn()

    def close(self):
        try:
            self.reset()
            self.update()
            self._ui.close()
        except Exception:
            pass


def _clamp16(v):
    return max(-32768, min(32767, int(v)))
