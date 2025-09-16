from main import read_all_data, load_taxonomy

if __name__ == "__main__":
    test_files_prefix = ['24292', '24322', '1462711', '42113', '46010', 'blctit1', 'blhpar1', 'blkvul']
    base_train_audio_path = "D://Dev//ML//workspace//kaggle_challenges//BirdCLEF_CHALLENGE//data//train_audio//"
    taxonomy_path = "D://Dev//ML//workspace//kaggle_challenges//BirdCLEF_CHALLENGE//data//taxonomy.csv"
    label_to_class = load_taxonomy(taxonomy_path)
    print(f"Label to class mapping: {label_to_class}")

    magnitude_df = read_all_data(base_train_audio_path, taxonomy_path, test_files_prefix,stop_percentage=0.03)
    # Resulting DataFrame
    print(magnitude_df.head())