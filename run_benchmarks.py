import os
import datetime
import subprocess

# Configuration
robot_types = ["cartken", "starship", "ottonomy", "nuro"]
datasets = [
    ("solomon_100.txt", "solomon100"),
    ("homberger_400.txt", "homberger400"),
    ("homberger_1000.TXT", "homberger1000")
]

# Get current date in DDMMYYYY format
current_date = datetime.datetime.now().strftime("%d%m%Y")

# Base output directory
output_base = "netro/output/"

# Run benchmarks
for robot in robot_types:
    for dataset_file, dataset_name in datasets:
        # Run Netro
        cmd = f"python -m netro.main --dataset {dataset_file} --robot {robot}"
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        
        # Find the latest output folder (last created)
        output_folders = sorted(
            [f for f in os.listdir(output_base) if os.path.isdir(os.path.join(output_base, f))],
            key=lambda f: os.path.getmtime(os.path.join(output_base, f)),
            reverse=True
        )
        
        if output_folders:
            latest_folder = output_folders[0]
            new_name = f"{current_date}_{dataset_name}_{robot}"
            old_path = os.path.join(output_base, latest_folder)
            new_path = os.path.join(output_base, new_name)
            
            # Remove existing directory if it exists
            if os.path.exists(new_path):
                import shutil
                shutil.rmtree(new_path)
            
            # Rename folder
            os.rename(old_path, new_path)
            print(f"Renamed '{latest_folder}' to '{new_name}'")
        else:
            print(f"No output folder found for {dataset_file} with {robot}")

print("All benchmarks completed!")
