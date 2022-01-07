import hydra
from omegaconf import DictConfig
from tsgrasp.data.data_generation.convert_meshes import convert_meshes
from tsgrasp.data.data_generation.add_contact_points import add_contact_points
from tsgrasp.data.data_generation.render_trajectories import render_trajectories
from shutil import copytree
import os

@hydra.main(config_path="../conf", config_name="scripts/generate_data")
def main(cfg : DictConfig):

    # print("####################################################")
    # print(f"Simplifying the meshes in {cfg.MESH_DIR} "
    #     f"and writing the results to {cfg.convert_meshes.OUTPUT_DIR} .")
    # convert_meshes(cfg.convert_meshes)

    print("####################################################")
    print(f"Generating grasp contact points.")
    # First, copy cfg.ACRONYM_DIR to cfg.OUTPUT_DATASET_DIR.
    # Then, add contact points to the h5 files in the new dataset.
    if not os.path.exists(cfg.OUTPUT_DATASET_DIR):
        copytree(cfg.ACRONYM_DIR, cfg.OUTPUT_DATASET_DIR)
    add_contact_points(cfg.add_contact_points)

    print("####################################################")
    print(f"Rendering trajectories.")
    render_trajectories(cfg.render_trajectories)
    
if __name__ == "__main__":
    main()