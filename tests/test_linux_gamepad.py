"""
Logic test for nitrogen.linux_gamepad.

Runs without a real /dev/uinput or python-evdev installed: it injects a tiny fake
`evdev` module that records every event written, using the real Linux kernel
event-code numbers. That makes the mapping assertions meaningful — it verifies
the gamepad translates NitroGen actions into the exact codes the kernel xpad
driver reports for a wired Xbox 360 pad, so Steam/Proton apply the right mapping.

Run: python tests/test_linux_gamepad.py
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- fake evdev with real kernel code numbers --------------------------------
EV_SYN, EV_KEY, EV_ABS = 0, 1, 3
CODES = dict(
    EV_SYN=EV_SYN, EV_KEY=EV_KEY, EV_ABS=EV_ABS,
    BTN_SOUTH=0x130, BTN_EAST=0x131, BTN_NORTH=0x133, BTN_WEST=0x134,
    BTN_TL=0x136, BTN_TR=0x137, BTN_SELECT=0x13A, BTN_START=0x13B,
    BTN_MODE=0x13C, BTN_THUMBL=0x13D, BTN_THUMBR=0x13E,
    ABS_X=0x00, ABS_Y=0x01, ABS_Z=0x02, ABS_RX=0x03, ABS_RY=0x04, ABS_RZ=0x05,
    ABS_HAT0X=0x10, ABS_HAT0Y=0x11,
)

fake = types.ModuleType("evdev")
ecodes = types.SimpleNamespace(**CODES)


class _AbsInfo:
    def __init__(self, value=0, min=0, max=0, fuzz=0, flat=0, resolution=0):
        self.min, self.max = min, max


class _UInput:
    last = None

    def __init__(self, cap=None, name=None, vendor=0, product=0, version=0):
        self.name, self.vendor, self.product = name, vendor, product
        self.events = []          # (type, code, value) since last syn
        self.frames = []          # snapshots per syn()
        _UInput.last = self

    def write(self, etype, code, value):
        self.events.append((etype, code, value))

    def syn(self):
        self.frames.append(dict(((c, v) for (_t, c, v) in self.events)))
        self.events = []

    def close(self):
        pass


fake.UInput = _UInput
fake.ecodes = ecodes
fake.AbsInfo = _AbsInfo
sys.modules["evdev"] = fake

import nitrogen.linux_gamepad as vg  # noqa: E402

failures = []


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        failures.append(name)


print("test_linux_gamepad:")

pad = vg.VX360Gamepad()
ui = _UInput.last
check("device advertises Xbox 360 vendor/product", ui.vendor == 0x045E and ui.product == 0x028E)

# Face-button mapping must match the real xpad codes.
check("A (SOUTH token) -> BTN_SOUTH", vg.XUSB_BUTTON.XUSB_GAMEPAD_A == 0x130)
check("B (EAST token)  -> BTN_EAST", vg.XUSB_BUTTON.XUSB_GAMEPAD_B == 0x131)
check("X (WEST token)  -> BTN_X(=NORTH)", vg.XUSB_BUTTON.XUSB_GAMEPAD_X == 0x133)
check("Y (NORTH token) -> BTN_Y(=WEST)", vg.XUSB_BUTTON.XUSB_GAMEPAD_Y == 0x134)

# Press A + set sticks/triggers, then flush one frame.
pad.reset()
pad.left_joystick(x_value=999999, y_value=-999999)   # over-range -> clamps to int16
pad.right_joystick(x_value=100, y_value=200)
pad.left_trigger(value=300)                           # over-range -> clamps to 255
pad.right_trigger(value=-5)                           # under-range -> clamps to 0
pad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
pad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
pad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
pad.update()
frame = ui.frames[-1]

check("left stick X clamps to +32767", frame[CODES["ABS_X"]] == 32767)
check("left stick Y clamps to -32768", frame[CODES["ABS_Y"]] == -32768)
check("right stick passes through", frame[CODES["ABS_RX"]] == 100 and frame[CODES["ABS_RY"]] == 200)
check("left trigger clamps to 255", frame[CODES["ABS_Z"]] == 255)
check("right trigger clamps to 0", frame[CODES["ABS_RZ"]] == 0)
check("A button pressed (=1)", frame[CODES["BTN_SOUTH"]] == 1)
check("B button not pressed (=0)", frame[CODES["BTN_EAST"]] == 0)
check("dpad LEFT -> HAT0X = -1", frame[CODES["ABS_HAT0X"]] == -1)
check("dpad UP   -> HAT0Y = -1", frame[CODES["ABS_HAT0Y"]] == -1)

# reset() must return everything to neutral on the next frame.
pad.reset()
pad.update()
frame = ui.frames[-1]
check("reset zeroes sticks", frame[CODES["ABS_X"]] == 0 and frame[CODES["ABS_Y"]] == 0)
check("reset zeroes triggers", frame[CODES["ABS_Z"]] == 0 and frame[CODES["ABS_RZ"]] == 0)
check("reset releases A", frame[CODES["BTN_SOUTH"]] == 0)
check("reset centers dpad", frame[CODES["ABS_HAT0X"]] == 0 and frame[CODES["ABS_HAT0Y"]] == 0)

print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED: {failures}'}")
sys.exit(1 if failures else 0)
