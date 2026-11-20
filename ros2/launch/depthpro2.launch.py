from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGS = {
    "model_repo": "christianengel/DepthPro2-S-ONNX",
    "model_file": "onnx/model_fp16.onnx",
    "model_path": "",
    "provider": "cuda",
    "image_topic": "/camera/image_raw",
    "camera_info_topic": "/camera/camera_info",
    "publish_pointcloud": "false",
    "pointcloud_stride": "2",
    "max_depth": "20.0",
}


def generate_launch_description():
    return LaunchDescription(
        [DeclareLaunchArgument(name, default_value=value) for name, value in ARGS.items()]
        + [
            Node(
                package="depthpro2_ros",
                executable="depthpro2_onnx_node",
                name="depthpro2",
                output="screen",
                parameters=[{name: LaunchConfiguration(name) for name in ARGS}],
            )
        ]
    )
