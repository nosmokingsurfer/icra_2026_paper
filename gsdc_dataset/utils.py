from pathlib import Path
from tqdm import tqdm

def get_tasks_for_dataset(data_path):
    imu_files = list(Path(data_path + "train/").rglob("**/device_imu.csv"))


    print("Indexing tasks...")
    tasks = []
    for t in tqdm(imu_files):
        sample_id = t.parts[-3].replace('-','_') + "_" + t.parts[-2]
        imu_file = str(t.parent / "device_imu.csv")
        gt_file = str(t.parent / "ground_truth.csv")

        tasks.append(
            {
                "sample_id" : sample_id,
                "mode" : "train",
                "imu_file" : imu_file,
                "gt_file" : gt_file
            }
        )

    return tasks
