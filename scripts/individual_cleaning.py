import os
import random
import librosa
import soundfile as sf
import shutil
import matplotlib.pyplot as plt
import numpy as np

# Check to make sure the files are formatted correctly
def check_and_fix_wave_format(input_folder, output_folder):
    problematic_files = []

    # Make sure there is an output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all the files in input folder
    for file in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)

        try:
            # Read the WAV file using librosa and convert to mono
            aud, Fs = librosa.load(input_path, sr=None, mono=True)

            # Save the mono audio to the output directory using soundfile.write
            sf.write(output_path, aud, Fs)

            print(f"Successfully processed {file} and saved to {output_path}.")

        # Save the list of problematic files so I can see where they came from
        except Exception as e:
            print(f"Error processing {file}: {e}")
            print("This file is problematic. Continuing with the next file.")
            problematic_files.append(file)

    return problematic_files

# Use an envelope to trim silence
def apply_envelope_and_trim(file_path):
    try:
        # Read the WAV file using librosa and convert to mono
        aud, Fs = librosa.load(file_path, sr=None, mono=True)

        # Apply an envelope to the mono audio signal
        aud_envelope, index = librosa.effects.trim(aud)

        # Save the trimmed audio to the same directory using soundfile.write
        sf.write(file_path, aud_envelope, Fs)

        print(f"Successfully processed {file_path}.")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        print("This file is problematic. Continuing with the next file.")

# Trim or pad the audio to 2 seconds
def trim_or_pad_audio(file_path, target_duration=2.0):
    try:
        # Read the WAV file using librosa and convert to mono
        aud, Fs = librosa.load(file_path, sr=None, mono=True)

        # Trim or pad the audio to the target duration
        if len(aud) < int(target_duration * Fs):
            aud = librosa.effects.preemphasis(aud, coef=0)  # Pad with silence
        else:
            aud = aud[:int(target_duration * Fs)]  # Trim to target duration

        # Save the trimmed or padded audio to the same directory using soundfile.write
        sf.write(file_path, aud, Fs)

        print(f"Successfully processed {file_path}.")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        print("This file is problematic. Continuing with the next file.")

# Normalize the audio 
def normalize_and_save(file_path):
    try:
        # Read the WAV file using librosa and convert to mono
        aud, Fs = librosa.load(file_path, sr=None, mono=True)

        # Normalize the audio signal
        aud_normalized = aud / max(abs(aud))

        # Save the normalized audio to the same directory using soundfile.write
        sf.write(file_path, aud_normalized, Fs)

        print(f"Successfully processed {file_path}.")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        print("This file is problematic. Continuing with the next file.")

# Augment the folder for each type of drum to get 1,000 total
def augment_and_save(folder_path, target_samples, prefix, starting_count=1):
    existing_samples_count, existing_samples = count_existing_samples(folder_path)

    # Determine how many additional samples are needed to reach the target
    additional_samples_needed = max(0, target_samples - existing_samples_count)

    # Randomly select existing samples to make copies and add noise to the copies
    samples_to_copy_and_add_noise = random.sample(existing_samples, additional_samples_needed)

    count = starting_count

    for sample in samples_to_copy_and_add_noise:
        original_path = os.path.join(folder_path, sample)

        # Make a copy of the original sample
        copy_path = os.path.join(folder_path, f"{prefix}_copy_{count}.wav")
        shutil.copyfile(original_path, copy_path)

        try:
            # Read the WAV file using librosa and convert to mono
            aud, Fs = librosa.load(copy_path, sr=None, mono=True)

            # Add 10% noise to the audio signal
            aud_with_noise = aud + 0.1 * random.uniform(-1, 1)

            # Save the noisy audio to the same directory using soundfile.write
            sf.write(copy_path, aud_with_noise, Fs)

            print(f"Successfully added noise to copy of {sample} and saved to {copy_path}.")

        except Exception as e:
            print(f"Error processing copy of {sample}: {e}")
            print("Continuing with the next file.")

        count += 1

    # Return the updated count for potential later use
    return count  

# Rename the files into a uniform format
def rename_files(folder_path, prefix, starting_count=1):
    count = starting_count

    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        new_file_path = os.path.join(folder_path, f"{prefix}_{count}.wav")
        os.rename(file_path, new_file_path)

        print(f"Renamed {file} to {new_file_path}.")
        count += 1

    return count  # Return the updated count for potential later use

# Count the number of existing samples in the folder
def count_existing_samples(folder_path):
    existing_samples = [file for file in os.listdir(folder_path) if file.endswith(".wav")]
    return len(existing_samples), existing_samples

# Plot the samples
def plot_samples(folder_path, num_samples=4):
    files = os.listdir(folder_path)

    for i in range(min(num_samples, len(files))):
        file_path = os.path.join(folder_path, files[i])
        aud, _ = librosa.load(file_path, sr=None, mono=True)
        
        plt.figure(figsize=(10, 10))
        plt.plot(aud)
        plt.title(f"Sample {i + 1}")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.show()
        
# Compress, take the spectrogram, compress, and save
def compress_and_save_spectrogram(file_path, time_compression_factor=1.0):
    # Load audio file
    audio, sr = librosa.load(file_path, sr=None)

    # Compute spectrogram
    S = librosa.feature.melspectrogram(y=audio, sr=sr)

    # Compress the time axis for spectrogram
    compressed_time_spectrogram = np.linspace(0, S.shape[1] - 1, int(S.shape[1] * time_compression_factor))
    compressed_time_spectrogram = np.round(compressed_time_spectrogram).astype(int)

    # Display compressed spectrogram without axes
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S[:, compressed_time_spectrogram], ref=np.max), x_axis='time', cmap='viridis', y_axis='mel', fmax=8000)
    plt.axis('off')  # Turn off axes
    
    # Get the file name and extension
    file_name, file_extension = os.path.splitext(os.path.basename(file_path))
    
    # Define the output image path with the same name as the audio file but with a different extension
    output_image_path = os.path.join(os.path.dirname(file_path), f"{file_name}.png")
    
    # Save as PNG
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)  
    plt.close()  
    
    os.remove(file_path)
    
    print(f"Successfully compressed and saved {file_name}.wav as the spectrogram {file_name}.png")


# Which drum type to process
sample_type = 'kik'
# sample_type = 'cym'
# sample_type = 'snr'
# sample_type = 'tom'

#automatically defined folders
input_folder_path = f'.\\sample_data\\{sample_type}'
output_folder_path = f'.\\sample_data\\processed_{sample_type}'

# This is so I can choose to process data or just plot it
process = True

if process:
    # Step 1: Check and fix wave format
    problematic_files = check_and_fix_wave_format(input_folder_path, output_folder_path)
    
    # Step 2: Apply envelope, trim silence, and trim or pad audio
    for file in os.listdir(output_folder_path):
        file_path = os.path.join(output_folder_path, file)
        apply_envelope_and_trim(file_path)
        trim_or_pad_audio(file_path, target_duration=5.0)
    
    # Step 3: Normalize and save
    for file in os.listdir(output_folder_path):
        file_path = os.path.join(output_folder_path, file)
        normalize_and_save(file_path)
    
    # Step 4: Augment and save
    current_count = augment_and_save(output_folder_path, target_samples=1000, prefix=sample_type)
    
    # Step 5: Rename files
    rename_files(output_folder_path, prefix=sample_type)
    
    # Step 6: Compress the data and save spectrogram
    for file in os.listdir(output_folder_path):
        file_path = os.path.join(output_folder_path, file)
        compress_and_save_spectrogram(file_path, time_compression_factor=0.5)
        
else: 
    plot_samples(output_folder_path, num_samples=1)