from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

import numpy as np
import threading
import time
from multiprocessing import Array, Value, Lock

import logging_mp

logger_mp = logging_mp.get_logger(__name__)

Inspire_Num_Motors = 6
kTopicInspireCommand = "rt/inspire/cmd"
kTopicInspireState = "rt/inspire/state"


class Inspire_Grip1_Controller:
    """Single-scalar per hand controller for Inspire hand.

    Maps a closure in [0,1] to Inspire's 6 normalized joints per hand (0=open, 1=closed internally maps to 1-open).
    """

    def __init__(
        self,
        left_in: Value,
        right_in: Value,
        data_lock: Lock,
        state_arr: Array,
        action_arr: Array,
        fps: float = 100.0,
        Unit_Test: bool = False,
        simulation_mode: bool = False,
    ):
        logger_mp.info("Initialize Inspire_Grip1_Controller...")
        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode
        self.data_lock = data_lock
        self.state_arr = state_arr
        self.action_arr = action_arr

        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0)

        self.HandCmb_publisher = ChannelPublisher(kTopicInspireCommand, MotorCmds_)
        self.HandCmb_publisher.Init()
        self.HandState_subscriber = ChannelSubscriber(kTopicInspireState, MotorStates_)
        self.HandState_subscriber.Init()

        self.left_state = Value('d', 0.0, lock=True)
        self.right_state = Value('d', 0.0, lock=True)

        self.subscribe_state_thread = threading.Thread(target=self._subscribe_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        # Initialize cmd msg
        self.hand_msg = MotorCmds_()
        self.hand_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(12)]

        # Control thread
        self.control_thread = threading.Thread(
            target=self._control_loop, args=(left_in, right_in), daemon=True
        )
        self.control_thread.start()
        logger_mp.info("Initialize Inspire_Grip1_Controller OK!\n")

    def _subscribe_state(self):
        while True:
            msg = self.HandState_subscriber.Read()
            if msg is not None:
                # Use index finger proximal joint as simple grip state proxy (normalize in [0,1])
                # Right hand index: id=3; Left hand index: id=9
                try:
                    with self.left_state.get_lock():
                        self.left_state.value = float(msg.states[9].q)
                    with self.right_state.get_lock():
                        self.right_state.value = float(msg.states[3].q)
                except Exception:
                    pass
                # Export to shared arrays
                if self.data_lock:
                    with self.data_lock:
                        self.state_arr[:] = [self.left_state.value, self.right_state.value]
            time.sleep(0.002)

    @staticmethod
    def _map_closure_to_inspire(closure: float) -> np.ndarray:
        """Map closure [0..1] (1=closed) to Inspire 6-joint normalized [0..1] (1=open).

        We invert so that 1 (closed) -> joints ~0 (closed), 0 (open) -> joints ~1 (open).
        Thumb yaw kept mid at 0.5.
        """
        closure = float(np.clip(closure, 0.0, 1.0))
        open_level = 1.0 - closure
        return np.array([open_level, open_level, open_level, open_level, open_level, 0.5], dtype=float)

    def _control_loop(self, left_in: Value, right_in: Value):
        try:
            while True:
                start = time.time()
                with left_in.get_lock():
                    l = float(left_in.value)
                with right_in.get_lock():
                    r = float(right_in.value)

                left_q = self._map_closure_to_inspire(l)
                right_q = self._map_closure_to_inspire(r)

                # Fill cmd message (right 0..5, left 6..11)
                for i in range(6):
                    self.hand_msg.cmds[i].q = right_q[i]
                for i in range(6):
                    self.hand_msg.cmds[6 + i].q = left_q[i]
                self.HandCmb_publisher.Write(self.hand_msg)

                if self.data_lock:
                    with self.data_lock:
                        self.action_arr[:] = [l, r]

                dt = time.time() - start
                time.sleep(max(0.0, (1.0 / self.fps) - dt))
        except Exception as e:
            logger_mp.error(f"Inspire_Grip1_Controller loop error: {e}")

