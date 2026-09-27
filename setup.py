from setuptools import setup, find_namespace_packages

setup(
    name="imu4d",
    version="0.1.0",
    description="4D human–scene understanding from wearable IMUs",
    packages=find_namespace_packages(
        include=[
            "models*",
            "training*",
            "utils*",
            "evaluation*",
            "metric*",
            "dataset_process*",
            "imu_synthesis*",
            "motion_tokenizer*",
            "visualize*",
        ]
    ),
    package_data={
        "dataset_process": ["*.json", "*.npz", "sample_data/*.pkl"],
        "motion_tokenizer": ["configs/*.json", "pretrained_weight/*.pth"],
    },
    python_requires=">=3.9",
)
