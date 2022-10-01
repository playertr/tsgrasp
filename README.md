# tsgrasp

TSGrasp: spatio-temporal grasping. Multiple frames go in, dense 6-DOF grasps come out.

[![Demo Video](https://img.youtube.com/vi/JBEWkCMrQKs/0.jpg)](https://www.youtube.com/watch?v=JBEWkCMrQKs)

# Installation
- Install CUDA 11
- `conda install mamba`
- `mamba env create -f environment.yaml`
- `conda activate tsgrasp`
- pip install from https://github.com/NVIDIA/MinkowskiEngine
- `mamba install -c conda-forge ros-rospy`
- install ros_numpy from local build with pip install .
- `pip install kornia`
- `pip install "git+https://github.com/facebookresearch/pytorch3d.git"`

# Usage
- `python -m train experiment=scene_random_yaw`
- `python -m scripts.save_outputs`