import soundfile as sf
import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed


def read_audio_file(audio_path, max_frequency=1_000):
    signal, fs = sf.read(audio_path)
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), d=1 / fs)
    if len(frequencies) > max_frequency:
        magnitude = np.abs(fft_result)[:max_frequency]
    else:
        magnitude = np.pad(np.abs(fft_result), (0, max_frequency - len(frequencies)), mode='constant')
    return magnitude


def load_taxonomy(taxonomy_path, columns=["primary_label"]):
    taxonomy = pd.read_csv(taxonomy_path).filter(items=columns)
    taxonomy['primary_label_code'] = taxonomy['primary_label'].astype('category').cat.codes
    return taxonomy.set_index("primary_label")["primary_label_code"].to_dict(),taxonomy['primary_label'].astype('category').cat.codes


def process_file(file_path, class_name_code, max_frequency=1_000):
    magnitude = read_audio_file(file_path, max_frequency)
    return {"class_name_code": class_name_code, "magnitude": magnitude}

def read_all_data(train_path, taxonomy_path, test_files_prefix, max_workers=2, stop_percentage=90):
    label_to_class,_ = load_taxonomy(taxonomy_path)
    data = []
    file_list = []

    for root, _, files in os.walk(train_path):
        folder_name = os.path.basename(root)
        if folder_name in label_to_class and folder_name not in test_files_prefix:
            class_name_code = label_to_class[folder_name]
            for file in files:
                file_path = os.path.join(root, file)
                file_list.append((file_path, class_name_code))

    total_files = len(file_list)
    stop_count = int((stop_percentage / 100) * total_files)

    print(f"Processing up to {stop_count} out of {total_files} files ({stop_percentage}%)...")

    # Submit and process incrementally
    # Using ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, (file_path, class_name_code) in enumerate(file_list):
            if i >= stop_count:
                break
            futures.append(executor.submit(process_file, file_path, class_name_code))

        for future in tqdm(as_completed(futures), total=stop_count, desc="Processing files"):
            data.append(future.result())


    print("\nProcessing complete.")

    df = pd.DataFrame(data)
    magnitude_df = pd.DataFrame(df["magnitude"].tolist())
    magnitude_df["class_name_code"] = df["class_name_code"]
    return magnitude_df
