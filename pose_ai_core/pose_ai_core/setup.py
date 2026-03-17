from setuptools import setup, find_packages

setup(
    name="pose-ai-core",
    version="1.0.0",
    description="Standalone pose estimation and exercise critique library",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.14.0",
        "opencv-python>=3.4.0.14",
        "torch>=0.4.1",
        "torchvision>=0.2.1",
    ],
    package_data={
        # include any .pth model checkpoint bundled with the package
        "pose_ai_core": ["*.pth"],
    },
)

