from read_data import TaskReader
import numpy as np

data_path = "/mnt/data_ssd/zhoufang/code/evo-fast/Evo_1/dataset/real_data/task_1"
reader = TaskReader(data_path)

ep = reader.get_episode(0)
step_data = ep.get_step_data(0)

print("Step Data Keys:", step_data.keys())
if "rgb" in step_data:
    print("RGB shape:", step_data["rgb"].shape)
    print("RGB dtype:", step_data["rgb"].dtype)
else:
    print("Warning: 'rgb' key not found in step_data!")
    print("Available keys:", list(step_data.keys())[:5])