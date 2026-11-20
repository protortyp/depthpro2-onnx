"""End-to-end check of the node: publish known frames, compare the depth it returns.

Publishes the 16 fixture frames upscaled to 1280x960 as bgr8, with intrinsics scaled to match,
one at a time, waiting for each depth image. The node must resize, convert colors and rescale
f_px correctly to reproduce the PyTorch reference: per-frame depth for single-frame graphs,
Apple's unbuffered stream for stream graphs (``--window``).

    python3 ros2/test/fake_camera_check.py fixtures/test_clip_480x640.npz \
        fixtures/ref_outputs/DepthPro2-S_480x640.npz [--window 4]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from depthpro2_onnx.metrics import compare, merge  # noqa: E402

SCALE = 2
QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)


class FakeCamera(Node):
    def __init__(self, f_px: float, shape: tuple[int, int]):
        super().__init__("fake_camera")
        h, w = shape[0] * SCALE, shape[1] * SCALE
        f = f_px * SCALE
        self.info = CameraInfo(width=w, height=h, distortion_model="plumb_bob")
        self.info.k = [f, 0.0, w / 2, 0.0, f, h / 2, 0.0, 0.0, 1.0]
        self.info.d = [0.0] * 5
        self.image_pub = self.create_publisher(Image, "/camera/image_raw", QOS)
        self.info_pub = self.create_publisher(CameraInfo, "/camera/camera_info", QOS)
        self.create_subscription(Image, "/depthpro2/depth", self.on_depth, QOS)
        self.create_subscription(CameraInfo, "/depthpro2/camera_info", self.on_info, QOS)
        self.create_subscription(PointCloud2, "/depthpro2/points", self.on_points, QOS)
        self.depths: dict[int, np.ndarray] = {}
        self.out_info: CameraInfo | None = None
        self.points = 0

    def publish(self, rgb: np.ndarray, stamp_ns: int) -> None:
        frame = np.ascontiguousarray(rgb.repeat(SCALE, axis=0).repeat(SCALE, axis=1)[..., ::-1])
        self.info.header.stamp.sec, self.info.header.stamp.nanosec = divmod(stamp_ns, 10**9)
        msg = Image(height=frame.shape[0], width=frame.shape[1], encoding="bgr8")
        msg.header.stamp = self.info.header.stamp
        msg.header.frame_id = "camera"
        msg.step = 3 * msg.width
        msg.data = frame.tobytes()
        self.info_pub.publish(self.info)
        self.image_pub.publish(msg)

    def on_depth(self, msg: Image) -> None:
        assert msg.encoding == "32FC1"
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        self.depths[stamp] = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)

    def on_info(self, msg: CameraInfo) -> None:
        self.out_info = msg

    def on_points(self, msg: PointCloud2) -> None:
        self.points = msg.width


def wait_for_depth(node: FakeCamera, stamp: int, timeout: float) -> np.ndarray | None:
    """The depth image stamped ``stamp``; depth for other frames is discarded."""
    deadline = time.time() + timeout
    while stamp not in node.depths and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.01)
    depth = node.depths.pop(stamp, None)
    node.depths.clear()
    return depth


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture")
    parser.add_argument("refs")
    parser.add_argument("--window", type=int, default=0, help="stream window; 0 = single frame")
    args = parser.parse_args()
    fixture = np.load(args.fixture)
    refs = np.load(args.refs)
    ref = refs[f"fp32/stream{args.window}/clip"] if args.window else refs["fp32/single/clip"]
    clip = fixture["clip"]

    rclpy.init()
    node = FakeCamera(float(fixture["f_px"]), clip.shape[1:3])
    # Warmup frames carry stamps far in the future, so the clip's earlier stamps make the node
    # reset its stream state and the clip is seen from its first frame, as in the reference.
    warmup = None
    for attempt in range(120):
        stamp = 10**18 + attempt
        node.publish(clip[0], stamp)
        warmup = wait_for_depth(node, stamp, 2.0)
        if warmup is not None:
            break
    if warmup is None:
        print("FAIL: node never answered")
        return 1

    results, latencies = [], []
    for i, rgb in enumerate(clip):
        start = time.perf_counter()
        # Best-effort QoS may drop a 3.7 MB image while the node is busy; resend it then.
        for _ in range(3):
            node.publish(rgb, (i + 1) * 10**8)
            depth = wait_for_depth(node, (i + 1) * 10**8, 10.0)
            if depth is not None:
                break
        if depth is None:
            print(f"FAIL: no depth for frame {i}")
            return 1
        latencies.append((time.perf_counter() - start) * 1e3)
        results.append(compare(depth[None], ref[i : i + 1]))
    result = merge(results)
    k = list(node.out_info.k) if node.out_info else None
    print(f"frames: {len(results)}, camera_info k: {k}, point cloud points: {node.points}")
    print(f"publish-to-depth latency median {np.median(latencies):.1f} ms (1280x960 bgr8 in)")
    print(
        f"vs PyTorch FP32 {'stream%d' % args.window if args.window else 'per frame'}: "
        f"AbsRel {result['abs_rel']:.5f} delta1 {result['delta1']:.5f}"
    )
    ok = result["abs_rel"] < 0.01 and result["delta1"] > 0.995
    print("PASS" if ok else "FAIL")
    node.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
