#!/usr/bin/env python3
"""wasd_teleop: hold-to-walk keyboard -> geometry_msgs/Twist on /cmd_vel.

TRUE press/release via evdev (raw /dev/input events -- terminals only see key
REPEATS, so 'stop when I let go of W' is impossible with stdin; this is why
evdev). Hold a key = command its preset velocity; release = publish 0 and the
NODE-SIDE governor ramps v_des down smoothly (0.08 m/s^2 slew) -- the client
stays dumb on purpose.

Keymap (user-locked 2026-07-03, validated envelope ONLY):
  W (hold)  vx = +0.15 m/s   (twin-validated robust forward)
  S (hold)  vx = -0.10 m/s   (twin-validated robust backward)
  A / D     vy = +/-0.08 m/s STRAFE -- forwarded but the node governor CLAMPS
            vy to 0 until the V2 lateral campaign passes (keys pre-wired)
  Q / E     reserved for YAW (V3) -- currently a no-op with a console note
  SPACE     zero latch: immediate 0 command, held keys ignored until all released
  ESC       publish zero Twist x3 and exit

Requirements: pip evdev + membership in the `input` group
(`sudo usermod -aG input $USER`, then re-login). Reads keys SYSTEM-WIDE from
the keyboard device -- terminal focus is NOT required (be aware while typing
elsewhere; SPACE/ESC still act).

Usage:
  python3 wasd_teleop.py [--device /dev/input/eventN] [--rate 20]
Chain: wasd_teleop -> /cmd_vel -> wss_cmd_bridge.py -> gRPC :10000 -> governor.
"""
import argparse
import select
import sys

import rclpy
from geometry_msgs.msg import Twist

try:
    import evdev
    from evdev import ecodes
except ImportError:
    sys.exit("needs python-evdev: pip install evdev  (and `input` group membership)")

VX_FWD, VX_BACK, VY_STRAFE = 0.15, -0.10, 0.08   # validated envelope presets


def pick_device(path):
    if path:
        return evdev.InputDevice(path)
    for p in evdev.list_devices():
        d = evdev.InputDevice(p)
        caps = d.capabilities().get(ecodes.EV_KEY, [])
        if ecodes.KEY_W in caps and ecodes.KEY_SPACE in caps:
            return d
    sys.exit("no keyboard-like /dev/input device found (are you in the `input` group?)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="/dev/input/eventN (default: auto)")
    ap.add_argument("--rate", type=float, default=20.0)
    args = ap.parse_args()

    dev = pick_device(args.device)
    print(f"[wasd] reading {dev.path} ({dev.name}); W/S drive, A/D strafe "
          f"(node-clamped to 0 until V2), SPACE stop-latch, ESC quit")

    rclpy.init()
    node = rclpy.create_node("wasd_teleop")
    pub = node.create_publisher(Twist, "/cmd_vel", 10)

    held = set()
    stop_latch = False

    def target():
        t = Twist()
        if stop_latch:
            return t
        if ecodes.KEY_W in held:
            t.linear.x = VX_FWD
        elif ecodes.KEY_S in held:
            t.linear.x = VX_BACK
        if ecodes.KEY_A in held:
            t.linear.y = +VY_STRAFE
        elif ecodes.KEY_D in held:
            t.linear.y = -VY_STRAFE
        return t

    period = 1.0 / args.rate
    try:
        while rclpy.ok():
            r, _, _ = select.select([dev.fd], [], [], period)
            if r:
                for ev in dev.read():
                    if ev.type != ecodes.EV_KEY or ev.value == 2:  # ignore autorepeat
                        continue
                    down = ev.value == 1
                    if ev.code == ecodes.KEY_ESC and down:
                        raise KeyboardInterrupt
                    if ev.code == ecodes.KEY_SPACE and down:
                        stop_latch = True
                        held.clear()
                        print("[wasd] SPACE -> zero latch (release all keys to re-arm)")
                        continue
                    if ev.code in (ecodes.KEY_Q, ecodes.KEY_E) and down:
                        print("[wasd] Q/E = yaw, reserved for V3 (no-op)")
                        continue
                    if ev.code in (ecodes.KEY_W, ecodes.KEY_A,
                                   ecodes.KEY_S, ecodes.KEY_D):
                        if down:
                            if not stop_latch:
                                held.add(ev.code)
                        else:
                            held.discard(ev.code)
                            if stop_latch and not held:
                                stop_latch = False
                                print("[wasd] re-armed")
            pub.publish(target())
    except KeyboardInterrupt:
        pass
    finally:
        import time
        for _ in range(3):
            pub.publish(Twist())
            time.sleep(0.05)
        node.destroy_node()
        rclpy.try_shutdown()
        print("[wasd] zeroed + exit")


if __name__ == "__main__":
    main()
