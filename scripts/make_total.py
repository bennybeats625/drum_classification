import os
import shutil
import random

# Shuffle the individual drum sound folders and merge them
def shuffle_and_merge_folders(input_folders, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of files from each input folder
    folder_files = [os.listdir(folder) for folder in input_folders]

    # Shuffle the files in each folder
    shuffled_files = [random.sample(files, len(files)) for files in folder_files]

    # Determine the maximum number of files in any folder
    max_files = max(len(files) for files in shuffled_files)

    # Merge the files in alternating order into the output folder
    for i in range(max_files):
        for j, folder in enumerate(input_folders):
            if i < len(shuffled_files[j]):
                input_path = os.path.join(folder, shuffled_files[j][i])
                output_path = os.path.join(output_folder, shuffled_files[j][i])
                shutil.copyfile(input_path, output_path)
                print(f"Copied {input_path} to {output_path}")

# Define the input folders
folder1_path = r'.\sample_data\processed_kik'
folder2_path = r'.\sample_data\processed_cym'
folder3_path = r'.\sample_data\processed_snr'
folder4_path = r'.\sample_data\processed_tom'
output_path = r'.\sample_data\baby_drm'

shuffle_and_merge_folders([folder1_path, folder2_path, folder3_path, folder4_path], output_path)
#shuffle_and_merge_folders([folder1_path, folder2_path], output_path)

