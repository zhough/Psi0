#!/usr/bin/env python3
"""Robot state monitor — READ-ONLY, does NOT drive the robot.

Subscribes to the Unitree DDS state topics and periodically prints the robot's
current pose / state information:
  - rt/lowstate          : 29 body joint q/dq/tau + IMU (hg LowState_)
  - rt/odommodestate     : base position / body height / velocity (SportModeState_)
  - rt/dex3/left/state   : left hand joint q (HandState_)
  - rt/dex3/right/state  : right hand joint q (HandState_)

It creates NO publishers and never writes rt/lowcmd — it only reads.

Usage:
  # make sure the DDS traffic goes over the NIC that is wired to the G1
  export CYCLONEDDS_URI='<CycloneDDS><Domain><General><NetworkInterfaceAddress>192.168.123.123</NetworkInterfaceAddress></General></Domain></CycloneDDS>'
  python real/deploy/print_robot_state.py --interval 1.0

  # or pass the interface name directly (SDK builds the CycloneDDS config):
  python real/deploy/print_robot_state.py --iface enP8p1s0 --interval 1.0
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Optional

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_


# 29 body joints in motor_state order (same as G1_29_BodyIndex)
BODY_JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]

HAND_JOINT_NAMES = {
    "left": ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"],
    "right": ["thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"],
}


def fmt_vec(v, fmt: str = "{:8.3f}") -> str:
    return "[" + " ".join(fmt.format(float(x)) for x in v) + "]"


def rad2deg(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float64) * 180.0 / math.pi


def print_body(lowstate: LowState_) -> None:
    print("-- body (29 joints) ------------------------------------------------")
    for i, name in enumerate(BODY_JOINT_NAMES):
        m = lowstate.motor_state[i]
        temp = m.temperature[0] if hasattr(m, "temperature") and len(m.temperature) else float("nan")
        print(
            f"  [{i:2d}] {name:20s} "
            f"q={m.q:8.4f}  dq={m.dq:8.4f}  tau={m.tau_est:8.3f}  T={temp:5.1f}"
        )


def print_hand(side: str, handstate: Optional[HandState_]) -> None:
    print(f"-- hand {side} -----------------------------------------------------")
    if handstate is None:
        print("  (no data)")
        return
    for i, name in enumerate(HAND_JOINT_NAMES[side]):
        if i >= len(handstate.motor_state):
            break
        m = handstate.motor_state[i]
        print(f"  {name:12s} q={m.q:8.4f}  dq={m.dq:8.4f}  tau={m.tau_est:8.3f}")


def print_imu(lowstate: LowState_) -> None:
    imu = lowstate.imu_state
    print("-- imu ---------------------------------------------------------------")
    print("  rpy(rad)      :", fmt_vec(imu.rpy))
    print("  rpy(deg)      :", fmt_vec(rad2deg(imu.rpy)))
    print("  quaternion    :", fmt_vec(imu.quaternion))
    print("  gyroscope     :", fmt_vec(imu.gyroscope))
    print("  accelerometer :", fmt_vec(imu.accelerometer))
    print("  mode_machine  :", int(lowstate.mode_machine), " tick:", int(lowstate.tick))


def print_odom(odom: Optional[SportModeState_]) -> None:
    print("-- odom / base --------------------------------------------------------")
    if odom is None:
        print("  (no data)")
        return
    print("  position      :", fmt_vec(odom.position))
    print("  body_height   :", f"{float(odom.body_height):8.3f}")
    print("  velocity      :", fmt_vec(odom.velocity))
    print("  yaw_speed     :", f"{float(odom.yaw_speed):8.3f}")
    print("  foot_force    :", fmt_vec(odom.foot_force, "{:6.0f}"))
    imu = odom.imu_state
    print("  odom imu rpy  :", fmt_vec(imu.rpy), " deg:", fmt_vec(rad2deg(imu.rpy)))
    print("  odom quat     :", fmt_vec(imu.quaternion))


def print_remote(lowstate: LowState_) -> None:
    raw = np.asarray(lowstate.wireless_remote, dtype=np.uint8)
    print("-- wireless remote (raw 40 bytes) -------------------------------------")
    print("  ", raw[:12].tobytes().hex(" "))


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only robot state monitor (no driving).")
    parser.add_argument("--interval", type=float, default=1.0, help="refresh interval in seconds (default 1.0)")
    parser.add_argument("--iface", type=str, default=None, help="network interface name for CycloneDDS (e.g. enP8p1s0)")
    parser.add_argument("--once", action="store_true", help="print once and exit")
    args = parser.parse_args()

    interval = max(0.2, args.interval)
    ChannelFactoryInitialize(0, networkInterface=args.iface)

    lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
    lowstate_sub.Init()
    odom_sub = ChannelSubscriber("rt/odommodestate", SportModeState_)
    odom_sub.Init()
    left_hand_sub = ChannelSubscriber("rt/dex3/left/state", HandState_)
    left_hand_sub.Init()
    right_hand_sub = ChannelSubscriber("rt/dex3/right/state", HandState_)
    right_hand_sub.Init()

    print(f"[monitor] subscribing to rt/lowstate, rt/odommodestate, rt/dex3/*/state  (interval={interval}s)")
    print("[monitor] READ-ONLY: no motor commands are sent. Ctrl+C to stop.")

    try:
        while True:
            lowstate = lowstate_sub.Read()
            odom = odom_sub.Read()
            left_hand = left_hand_sub.Read()
            right_hand = right_hand_sub.Read()

            print("\n" + "=" * 68)
            print(f"time: {time.strftime('%H:%M:%S')}")
            if lowstate is None:
                print("[monitor] waiting for rt/lowstate ... (is the G1 on and DDS reachable?)")
            else:
                print_imu(lowstate)
                print_body(lowstate)
                print_remote(lowstate)
            print_odom(odom)
            print_hand("left", left_hand)
            print_hand("right", right_hand)
            print("=" * 68, flush=True)

            if args.once:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[monitor] stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
