from mask_generation import process_all_parallel



if __name__ == "__main__":
    train_root_path = r"../data/train"  # change if needed
    process_all_parallel(train_root_path, limit=5,print_overlay=True,thickness=2, workers=2)