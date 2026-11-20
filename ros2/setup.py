from setuptools import setup

package = "depthpro2_ros"

setup(
    name=package,
    version="0.1.0",
    packages=["depthpro2_onnx"],
    py_modules=["depthpro2_onnx_node"],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package}"]),
        (f"share/{package}", ["package.xml"]),
        (f"share/{package}/launch", ["launch/depthpro2.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    license="Apache-2.0",
    entry_points={"console_scripts": ["depthpro2_onnx_node = depthpro2_onnx_node:main"]},
)
