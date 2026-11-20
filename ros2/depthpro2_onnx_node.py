"""ROS 2 node: metric depth from a calibrated camera with a focal-free Depth Pro 2 ONNX graph.

Subscribes:  image (sensor_msgs/Image), camera_info (sensor_msgs/CameraInfo)
Publishes:   ~/depth (32FC1, meters), ~/camera_info (intrinsics of ~/depth), ~/points (optional)

The focal length comes from CameraInfo.k[0] (fx), so the Depth Pro 2 focal-length network is not
needed. Frames are scaled to cover the graph's input size and center-cropped; ~/camera_info
carries the matching intrinsics.
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField

from depthpro2_onnx.runtime import DepthPro2ONNX, to_pixel_values

ENCODINGS = {"rgb8": (3, False), "bgr8": (3, True), "rgba8": (4, False), "bgra8": (4, True)}


def image_to_rgb(msg: Image) -> np.ndarray:
    if msg.encoding == "mono8":
        gray = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, : msg.width]
        return np.repeat(gray[..., None], 3, axis=2)
    if msg.encoding not in ENCODINGS:
        raise ValueError(f"unsupported encoding {msg.encoding!r}; use rgb8, bgr8, rgba8 or mono8")
    channels, bgr = ENCODINGS[msg.encoding]
    rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    pixels = rows[:, : msg.width * channels].reshape(msg.height, msg.width, channels)[..., :3]
    return pixels[..., ::-1] if bgr else pixels


def resize(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    except ImportError:
        from PIL import Image as PILImage

        return np.asarray(PILImage.fromarray(rgb).resize((width, height), PILImage.BILINEAR))


def cover_crop(rgb: np.ndarray, k: np.ndarray, height: int, width: int):
    """Scale ``rgb`` to cover ``height`` x ``width``, center-crop, adjust intrinsics ``k``."""
    h, w = rgb.shape[:2]
    scale = max(width / w, height / h)
    sw, sh = max(width, round(w * scale)), max(height, round(h * scale))
    left, top = (sw - width) // 2, (sh - height) // 2
    if (sw, sh) != (w, h):
        rgb = resize(rgb, sw, sh)
    rgb = rgb[top : top + height, left : left + width]
    sx, sy = sw / w, sh / h
    k = k.copy()
    k[0, 0] *= sx
    k[1, 1] *= sy
    k[0, 2] = k[0, 2] * sx - left
    k[1, 2] = k[1, 2] * sy - top
    return np.ascontiguousarray(rgb), k


def point_cloud(depth: np.ndarray, k: np.ndarray, stride: int, max_depth: float, header):
    z = depth[::stride, ::stride]
    v, u = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    keep = np.isfinite(z) & (z > 0) & (z <= max_depth)
    z, u, v = z[keep], u[keep], v[keep]
    x = (u - k[0, 2]) * z / k[0, 0]
    y = (v - k[1, 2]) * z / k[1, 1]
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    msg = PointCloud2()
    msg.header = header
    msg.height, msg.width = 1, len(xyz)
    msg.fields = [
        PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
        for i, n in enumerate("xyz")
    ]
    msg.is_bigendian, msg.point_step, msg.row_step = False, 12, 12 * len(xyz)
    msg.is_dense = True
    msg.data = xyz.tobytes()
    return msg


class DepthPro2Node(Node):
    def __init__(self):
        super().__init__("depthpro2")
        p = self.declare_parameter
        model_path = p("model_path", "").value
        repo = p("model_repo", "christianengel/DepthPro2-S-ONNX").value
        filename = p("model_file", "onnx/model_fp16.onnx").value
        revision = p("model_revision", "main").value
        provider = p("provider", "cuda").value
        self.publish_points = p("publish_pointcloud", False).value
        self.stride = int(p("pointcloud_stride", 2).value)
        self.max_depth = float(p("max_depth", 20.0).value)
        image_topic = p("image_topic", "/camera/image_raw").value
        info_topic = p("camera_info_topic", "/camera/camera_info").value
        self.log_every = int(p("log_timing_every", 300).value)

        if not model_path:
            from huggingface_hub import hf_hub_download

            hf_hub_download(repo, "config.json", revision=revision)
            model_path = hf_hub_download(repo, filename, revision=revision)
        self.graph = DepthPro2ONNX(model_path, provider)
        self.get_logger().info(
            f"{model_path} on {self.graph.session.get_providers()[0]}, "
            f"input {self.graph.height}x{self.graph.width}, "
            f"{'stream' if self.graph.is_stream else 'single frame'}"
        )
        if self.graph.frames != 1:
            raise ValueError("Use a single-frame or stream graph; clip graphs take several frames.")

        self.info: CameraInfo | None = None
        self.last_stamp = None
        self.timing = np.zeros(4)
        self.timed_frames = 0
        self.warned_distortion = False
        self.create_subscription(CameraInfo, info_topic, self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image, image_topic, self.on_image, qos_profile_sensor_data)
        self.depth_pub = self.create_publisher(Image, "~/depth", qos_profile_sensor_data)
        self.info_pub = self.create_publisher(CameraInfo, "~/camera_info", qos_profile_sensor_data)
        self.points_pub = self.create_publisher(PointCloud2, "~/points", qos_profile_sensor_data)

    def on_info(self, msg: CameraInfo) -> None:
        if self.info is not None and (msg.width, msg.height) != (self.info.width, self.info.height):
            self.graph.reset()
        if not self.warned_distortion and any(abs(d) > 1e-6 for d in msg.d):
            self.get_logger().warn(
                "camera_info has distortion; subscribe to a rectified image (image_rect) so that "
                "the pinhole focal length holds."
            )
            self.warned_distortion = True
        self.info = msg

    def on_image(self, msg: Image) -> None:
        if self.info is None:
            self.get_logger().warn("waiting for camera_info", throttle_duration_sec=5.0)
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_stamp is not None and stamp < self.last_stamp:
            # Time went backwards (bag loop, camera restart): the cached frames are stale.
            self.graph.reset()
        self.last_stamp = stamp

        t0 = time.perf_counter()
        k = np.array(self.info.k, dtype=np.float64).reshape(3, 3)
        if (msg.width, msg.height) != (self.info.width, self.info.height):
            # Image and camera_info at different resolutions, e.g. a binned stream.
            k[0] *= msg.width / self.info.width
            k[1] *= msg.height / self.info.height
        rgb, k = cover_crop(image_to_rgb(msg), k, self.graph.height, self.graph.width)
        pixels = to_pixel_values(rgb)
        t1 = time.perf_counter()
        depth = self.graph.infer(pixels, float(k[0, 0]))[0]
        t2 = time.perf_counter()

        out = Image()
        out.header = msg.header
        out.height, out.width = depth.shape
        out.encoding, out.is_bigendian, out.step = "32FC1", 0, 4 * depth.shape[1]
        out.data = depth.astype(np.float32).tobytes()
        self.depth_pub.publish(out)

        info = CameraInfo()
        info.header = msg.header
        info.height, info.width = depth.shape
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        info.k = k.flatten().tolist()
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [k[0, 0], 0.0, k[0, 2], 0.0, 0.0, k[1, 1], k[1, 2], 0.0, 0.0, 0.0, 1.0, 0.0]
        self.info_pub.publish(info)

        t3 = time.perf_counter()
        if self.publish_points:
            self.points_pub.publish(point_cloud(depth, k, self.stride, self.max_depth, msg.header))
        t4 = time.perf_counter()

        self.timing += np.diff([t0, t1, t2, t3, t4]) * 1e3
        self.timed_frames += 1
        if self.log_every and self.timed_frames == self.log_every:
            pre, model, depth_out, points = self.timing / self.timed_frames
            self.get_logger().info(
                f"ms/frame: preprocess {pre:.1f}, model {model:.1f}, publish depth {depth_out:.1f}"
                f", point cloud {points:.1f}"
            )
            self.timing[:] = 0
            self.timed_frames = 0


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DepthPro2Node()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
