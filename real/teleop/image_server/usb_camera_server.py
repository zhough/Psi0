#!/usr/bin/env python3
"""Serve G1 built-in HDR (UVC) camera frames over ZMQ.

Compatible with image_client.py: replies to any request with
send_multipart([rgb_jpg, ir_jpg, depth_bytes]).

The G1 built-in cameras have no IR and no depth, so:
  - ir_jpg    = empty bytes (client skips the IR window)
  - depth     = fake zeros 640x480 uint16 (keeps the wire protocol intact)

Usage (on the robot):
    python usb_camera_server.py --index 0
    python usb_camera_server.py --index 2 --port 5558   # second camera
"""

import argparse

import cv2
import numpy as np
import zmq

FAKE_DEPTH_BYTES = np.zeros((480, 640), dtype=np.uint16).tobytes()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve G1 USB HDR camera over ZMQ")
    parser.add_argument("--index", type=int, default=0, help="/dev/videoN index (default: 0)")
    parser.add_argument("--ip", default="192.168.123.164", help="IP to bind (robot IP)")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ port (default: 5556)")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    parser.add_argument("--fps", type=int, default=30, help="Requested capture FPS (default: 30)")
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality 1-100 (default: 80)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.index)
    if not cap.isOpened():
        print(f"Failed to open camera at index {args.index} (/dev/video{args.index})")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(
        f"USB camera /dev/video{args.index}: "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
        f"@ {cap.get(cv2.CAP_PROP_FPS):.0f} FPS"
    )

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{args.ip}:{args.port}")
    print(f"ZMQ server on tcp://{args.ip}:{args.port}. Ctrl+C to stop.")

    try:
        while True:
            socket.recv()  # any request (image_client sends b"get")
            ret, frame = cap.read()
            if not ret or frame is None:
                socket.send(b"")
                continue

            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            if not ok:
                socket.send(b"")
                continue

            socket.send_multipart([encoded.tobytes(), b"", FAKE_DEPTH_BYTES])
    finally:
        cap.release()
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
