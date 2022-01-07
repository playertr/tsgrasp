## Generate data

1. Download object mesh data.
Download the ShapeNetSem [dataset](https://shapenet.org/). You have to create an account. You want `models-OBJ.zip`. Unzip it. Put it into `data/raw_datasets`. When you're done, `data/raw_datasets/acronym/grasps/` should be a folder full of `.h5` files.

2. Download ACRONYM grasp pose annotations.
Download `acronym.tar.gz` from the ACRONYM Github [page](https://github.com/NVlabs/acronym). Unzip it. Put it into `data/raw_datasets`. When you're done, `data/raw_datasets/models-OBJ/models` should be a folder full of `.obj` files.

3. Install the postprocessing tools, Manifold and Simplify.
Follow the instructions on the ACRONYM Github [page](https://github.com/NVlabs/acronym) to download and compile the Manifold [software](https://github.com/hjwdzh/Manifold) for mesh processing.

4. Create a conda environment with the data generation dependencies.
```
conda env create -f tsgrasp/data/data_generation/gen_data_env.yml
```

5. Update the configuration file with the locations 
