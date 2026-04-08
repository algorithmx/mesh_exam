"""
Mesh Issue Examiner MCP Framework

A comprehensive framework for automated mesh issue identification and quality assessment.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mesh-issue-examiner",
    version="0.1.0",
    author="MiniMax Agent",
    author_email="",
    description="MCP framework for mesh issue identification and quality assessment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/matrix/mesh-issue-examiner",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "mcp>=1.0.0",
    ],
    extras_require={
        "trimesh": ["trimesh>=4.0.0"],
        "pyvista": ["pyvista>=0.40.0", "vtk>=9.2.0"],
        "pymeshlab": ["pymeshlab>=2022.2"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "all": [
            "trimesh>=4.0.0",
            "pyvista>=0.40.0",
            "vtk>=9.2.0",
            "pymeshlab>=2022.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "mesh-examiner=mesh_issue_examiner.mcp.server:main",
        ],
    },
)
