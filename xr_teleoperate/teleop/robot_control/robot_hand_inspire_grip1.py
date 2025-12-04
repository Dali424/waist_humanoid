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
    def __init__(
        self,
        left_value_in: Value,
        right_value_in: Value,
        dual_data_lock: Lock,
        dual_state_out: Array,
        dual_action_out: Array,
        fps: float = 100.0,
        Unit_Test: bool = False,
        simulation_mode: bool = False,
    ):
        logger_mp.info("Initialize Inspire_Grip1_Controller...")
        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode
        self.data_lock = dual_data_lock
        self.state_arr = dual_state_out
        self.action_arr = dual_action_out

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

        self.hand_msg = MotorCmds_()
        self.hand_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(12)]

        self.ctrl_thread = threading.Thread(target=self._loop, args=(left_value_in, right_value_in), daemon=True)
        self.ctrl_thread.start()
        logger_mp.info("Initialize Inspire_Grip1_Controller OK!\n")

    def _subscribe_state(self):
        while True:
            msg = self.HandState_subscriber.Read()
            if msg is not None:
                try:
                    with self.left_state.get_lock():
                        self.left_state.value = float(msg.states[9].q)
                    with self.right_state.get_lock():
                        self.right_state.value = float(msg.states[3].q)
                except Exception:
                    pass
                if self.data_lock:
                    with self.data_lock:
                        self.state_arr[:] = [self.left_state.value, self.right_state.value]
            time.sleep(0.002)

    @staticmethod
    def _map_closure_to_inspire(closure: float) -> np.ndarray:
        closure = float(np.clip(closure, 0.0, 1.0))
        open_level = 1.0 - closure
        return np.array([open_level, open_level, open_level, open_level, open_level, 0.5], dtype=float)

    def _loop(self, left_value_in: Value, right_value_in: Value):
        try:
            while True:
                t0 = time.time()
                with left_value_in.get_lock():
                    l_raw = float(left_value_in.value)
                with right_value_in.get_lock():
                    r_raw = float(right_value_in.value)

                # TeleVuer pinch values are in [0, 100], where 0 ≈ fingers touching (strong pinch),
                # and larger values mean fingers are farther apart. Map to closure in [0, 1] so that:
                #   - closure = 1.0 when fingers are fully touching  -> hand fully closed
                #   - closure = 0.0 when fingers are far apart       -> hand fully open
                l_norm = np.clip(l_raw / 100.0, 0.0, 1.0)
                r_norm = np.clip(r_raw / 100.0, 0.0, 1.0)
                l_closure = 1.0 - l_norm
                r_closure = 1.0 - r_norm

                lq = self._map_closure_to_inspire(l_closure)
                rq = self._map_closure_to_inspire(r_closure)
                for i in range(6):
                    self.hand_msg.cmds[i].q = rq[i]
                for i in range(6):
                    self.hand_msg.cmds[6 + i].q = lq[i]
                self.HandCmb_publisher.Write(self.hand_msg)

                if self.data_lock:
                    with self.data_lock:
                        # Log the normalized closure (0=open, 1=closed) as the action.
                        self.action_arr[:] = [l_closure, r_closure]
                dt = time.time() - t0
                time.sleep(max(0.0, (1.0 / self.fps) - dt))
        except Exception as e:
            logger_mp.error(f"Inspire_Grip1 loop error: {e}")
