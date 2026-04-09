from setuptools import setup, find_namespace_packages

setup(
    name="imu4d",
    version="0.1.0",
    description="4D human–scene understanding from wearable IMUs",
    packages=find_namespace_packages(
        include=["models*", "training*", "utils*", "evaluation*", "dataset_process*", "imu_synthesis*"]
    ),
    python_requires=">=3.9",
)
