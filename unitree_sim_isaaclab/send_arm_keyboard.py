#!/usr/bin/env python3
"""
Keyboard teleop for G1 arms + waist pitch via DDS (rt/lowcmd)

Modes:
  1) Joint mode (default)
     - Left arm:  WASD
         - W/S: increase/decrease current left-arm joint
         - A/D: select previous/next left-arm joint (0..6)
     - Right arm: IJKL
         - I/K: increase/decrease current right-arm joint
         - J/L: select previous/next right-arm joint (0..6)
     - Waist: V forward, N backward  (pitch)

  2) Cartesian mode (simple IK approximation)
     - Left arm:  W/S -> +X/-X, A/D -> -Y/+Y, E/C -> +Z/-Z
     - Right arm: I/K -> +X/-X, J/L -> -Y/+Y, O/U -> +Z/-Z
     (delta_q = alpha * J_approx^T * delta_x, with tunable gains)

Common:
  - M: toggle joint/cartesian mode
  - R: re-sync targets to current lowstate
  - Q: quit
"""

import threading
import time
from typing import List
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

try:
    from pynput import keyboard
except Exception as e:
    print("pynput is required. Install with: pip install pynput")
    raise


ARM_BASE_IDX = 15
LEFT_SIZE = 7
RIGHT_SIZE = 7
WAIST_PITCH_IDX = 14

LEFT_NAMES = [
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
]
RIGHT_NAMES = [
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]


class ArmTeleop:
    def __init__(self, step=0.02, limit=2.5, rate_hz=50,
                 cart_step=0.01, cart_gain=0.5):
        self.step = float(step)
        self.limit = float(limit)
        self.dt = 1.0 / float(rate_hz)
        self.cart_step = float(cart_step)
        self.cart_gain = float(cart_gain)

        self._pos = [0.0] * 29
        self._last_lowstate = [0.0] * 29
        self._lock = threading.Lock()
        self._got_state = threading.Event()

        self.cartesian_mode = False
        self.left_sel = 0
        self.right_sel = 0

        self._keys = {
            # Left arm
            'w': False, 's': False, 'a': False, 'd': False,
            # Right arm
            'i': False, 'k': False, 'j': False, 'l': False,
            # Cartesian extras
            'e': False, 'c': False,
            'o': False, 'u': False,
            # Waist
            'v': False, 'n': False,  # v: forward bend, n: backward
        }
        self._running = True

        self._crc = CRC()
        self.pub = None
        self.sub = None

    def setup_dds(self):
        ChannelFactoryInitialize(1)

        def on_lowstate(msg: LowState_):
            try:
                with self._lock:
                    n = min(29, len(msg.motor_state))
                    for i in range(n):
                        v = float(msg.motor_state[i].q)
                        self._pos[i] = v if not self._got_state.is_set() else self._pos[i]
                        self._last_lowstate[i] = v
                self._got_state.set()
            except Exception:
                pass

        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(lambda m: on_lowstate(m), 32)

        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()

    def _on_press(self, key):
        try:
            ch = key.char.lower() if hasattr(key, 'char') and key.char else ''
        except Exception:
            return
        if ch in self._keys:
            self._keys[ch] = True
        elif ch == 'q':
            print("[teleop] quit requested")
            self._running = False
            return False
        elif ch == 'r':
            with self._lock:
                self._pos[:] = self._last_lowstate[:]
            print("[teleop] re-synced targets to current lowstate")
        elif ch == 'm':
            self.cartesian_mode = not self.cartesian_mode
            mode = "CARTESIAN" if self.cartesian_mode else "JOINT"
            print(f"[mode] switched to {mode} mode")
        if not self.cartesian_mode and ch == 'a':
            self.left_sel = (self.left_sel - 1) % LEFT_SIZE
            print(f"[left] select {self.left_sel}: {LEFT_NAMES[self.left_sel]}")
        elif not self.cartesian_mode and ch == 'd':
            self.left_sel = (self.left_sel + 1) % LEFT_SIZE
            print(f"[left] select {self.left_sel}: {LEFT_NAMES[self.left_sel]}")
        elif not self.cartesian_mode and ch == 'j':
            self.right_sel = (self.right_sel - 1) % RIGHT_SIZE
            print(f"[right] select {self.right_sel}: {RIGHT_NAMES[self.right_sel]}")
        elif not self.cartesian_mode and ch == 'l':
            self.right_sel = (self.right_sel + 1) % RIGHT_SIZE
            print(f"[right] select {self.right_sel}: {RIGHT_NAMES[self.right_sel]}")

    def _on_release(self, key):
        try:
            ch = key.char.lower() if hasattr(key, 'char') and key.char else ''
        except Exception:
            return
        if ch in self._keys:
            self._keys[ch] = False

    def start_keyboard(self):
        listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        listener.daemon = True
        listener.start()
        return listener

    def _clamp(self, v: float) -> float:
        return max(-self.limit, min(self.limit, v))

    def _apply_keys(self):
        if not self.cartesian_mode:
            if self._keys['w'] or self._keys['s']:
                idx = ARM_BASE_IDX + self.left_sel
                delta = self.step * (1.0 if self._keys['w'] else -1.0)
                with self._lock:
                    self._pos[idx] = self._clamp(self._pos[idx] + delta)
            if self._keys['i'] or self._keys['k']:
                idx = ARM_BASE_IDX + LEFT_SIZE + self.right_sel
                delta = self.step * (1.0 if self._keys['i'] else -1.0)
                with self._lock:
                    self._pos[idx] = self._clamp(self._pos[idx] + delta)
        else:
            dlx = dly = dlz = drx = dry = drz = 0.0
            if self._keys['w']: dlx += self.cart_step
            if self._keys['s']: dlx -= self.cart_step
            if self._keys['a']: dly -= self.cart_step
            if self._keys['d']: dly += self.cart_step
            if self._keys['e']: dlz += self.cart_step
            if self._keys['c']: dlz -= self.cart_step
            if self._keys['i']: drx += self.cart_step
            if self._keys['k']: drx -= self.cart_step
            if self._keys['j']: dry -= self.cart_step
            if self._keys['l']: dry += self.cart_step
            if self._keys['o']: drz += self.cart_step
            if self._keys['u']: drz -= self.cart_step
            if dlx or dly or dlz:
                dq_l = self._cartesian_to_joint_left(dlx, dly, dlz)
                with self._lock:
                    for i, dq in enumerate(dq_l):
                        self._pos[ARM_BASE_IDX + i] = self._clamp(self._pos[ARM_BASE_IDX + i] + dq)
            if drx or dry or drz:
                dq_r = self._cartesian_to_joint_right(drx, dry, drz)
                with self._lock:
                    for i, dq in enumerate(dq_r):
                        self._pos[ARM_BASE_IDX + LEFT_SIZE + i] = self._clamp(
                            self._pos[ARM_BASE_IDX + LEFT_SIZE + i] + dq)

        # Waist pitch
        if self._keys['v']:
            with self._lock:
                self._pos[WAIST_PITCH_IDX] = self._clamp(self._pos[WAIST_PITCH_IDX] + self.step)
        if self._keys['n']:
            with self._lock:
                self._pos[WAIST_PITCH_IDX] = self._clamp(self._pos[WAIST_PITCH_IDX] - self.step)

    def _cartesian_to_joint_left(self, dx, dy, dz) -> List[float]:
        w_x = [+0.7, 0.0, 0.0, -0.7, 0.0, -0.3, 0.0]
        w_y = [0.0, +0.8, +0.2, 0.0, +0.3, 0.0, +0.3]
        w_z = [-0.5, 0.0, +0.4, -0.4, 0.0, -0.2, 0.0]
        return [self.cart_gain * (dx*wx + dy*wy + dz*wz) for wx, wy, wz in zip(w_x, w_y, w_z)]

    def _cartesian_to_joint_right(self, dx, dy, dz) -> List[float]:
        w_x = [+0.7, 0.0, 0.0, -0.7, 0.0, -0.3, 0.0]
        w_y = [0.0, -0.8, -0.2, 0.0, -0.3, 0.0, -0.3]
        w_z = [-0.5, 0.0, -0.4, -0.4, 0.0, -0.2, 0.0]
        return [self.cart_gain * (dx*wx + dy*wy + dz*wz) for wx, wy, wz in zip(w_x, w_y, w_z)]

    def _build_lowcmd(self) -> unitree_hg_msg_dds__LowCmd_:
        msg = unitree_hg_msg_dds__LowCmd_()
        with self._lock:
            for i in range(29):
                mc = msg.motor_cmd[i]
                mc.q = float(self._pos[i])
                mc.dq = 0.0
                mc.tau = 0.0
                mc.kp = 0.0
                mc.kd = 0.0
        msg.mode_pr = 0
        msg.mode_machine = 0
        msg.crc = self._crc.Crc(msg)
        return msg

    def run(self):
        print("Waiting for initial lowstate...")
        self._got_state.wait(timeout=3.0)
        print("Starting teleop loop")
        while self._running:
            try:
                self._apply_keys()
                msg = self._build_lowcmd()
                self.pub.Write(msg)
            except Exception as e:
                print(f"[teleop] publish error: {e}")
            time.sleep(self.dt)


def main():
    print("=" * 60)
    print("Arm + Waist keyboard teleop")
    print("Modes:")
    print("- JOINT: Left A/D select, W/S move | Right J/L select, I/K move")
    print("- CART:  Left W/S=+/-X, A/D=-/+Y, E/C=+/-Z | Right I/K=+/-X, J/L=-/+Y, O/U=+/-Z")
    print("- Waist: V forward, N backward")
    print("Keys: M toggle mode, R resync, Q quit")
    print("=" * 60)

    teleop = ArmTeleop(step=0.02, limit=2.5, rate_hz=50, cart_step=0.01, cart_gain=0.5)
    teleop.setup_dds()
    listener = teleop.start_keyboard()
    try:
        teleop.run()
    finally:
        if listener:
            listener.stop()
        print("Exiting arm teleop")


if __name__ == "__main__":
    main()
