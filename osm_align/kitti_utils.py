import os
import numpy as np

def get_kitti_sequence_info(seq_id):
    """
    Given a KITTI sequence id (as int or str), return:
      - date (str)
      - drive (str)
      - [start_frame, end_frame] (list of int)
    """
    # Table of sequences
    kitti_sequences = {
        "00": ("2011_10_03", "0027", [0, 4540]),
        "01": ("2011_10_03", "0042", [0, 1100]),
        "02": ("2011_10_03", "0034", [0, 4660]),
        "03": ("2011_09_26", "0067", [0, 800]),
        "04": ("2011_09_30", "0016", [0, 270]),
        "05": ("2011_09_30", "0018", [0, 2760]),
        "06": ("2011_09_30", "0020", [0, 1100]),
        "07": ("2011_09_30", "0027", [0, 1100]),
        "08": ("2011_09_30", "0028", [1100, 5170]),
        "09": ("2011_09_30", "0033", [0, 1590]),
        "10": ("2011_09_30", "0034", [0, 1200]),
    }
    seq_id_str = str(seq_id).zfill(2)
    if seq_id_str not in kitti_sequences:
        raise ValueError(f"Unknown KITTI sequence id: {seq_id}")
    date, drive, frames = kitti_sequences[seq_id_str]
    return date, drive, frames

def odom_pose(odom_dir):
    if not os.path.exists(odom_dir):
        raise FileNotFoundError(f"Odom directory not found: {odom_dir}, Ground truth(Odometry) is available for only 10 sequences in KITTI. Stopping the process.")
    with open(odom_dir, 'r') as file:
        lines = file.readlines()
    transformation_data = [[float(val) for val in line.split()] for line in lines]
    homogenous_matrix_arr = []
    for i in range(len(transformation_data)):
        homogenous_matrix = np.identity(4)
        homogenous_matrix[0, :] = transformation_data[i][0:4]
        homogenous_matrix[1:2, :] = transformation_data[i][4:8]
        homogenous_matrix[2:3, :] = transformation_data[i][8:12]
        homogenous_matrix_arr.append(homogenous_matrix)
    return np.array(homogenous_matrix_arr)

def get_pose(path):
    poses=[]
    with open(path,"r") as f:
        lines=f.readlines()
    for line in lines:
        aux=[float(x) for x in line.split()]
        aux=np.array(aux).reshape(3,4)
        aux=np.vstack([aux,[0,0,0,1]])
        poses.append(aux)
    return np.array(poses)