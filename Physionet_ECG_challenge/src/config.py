from mask_generation import build_physical_template, compute_baseline_rows



H_T = 864
W_T = 1120
total_height_mm = 215.0
total_width_mm = 280.0
PX_PER_MM_Y = H_T / float(total_height_mm)
PX_PER_MM_X = W_T / float(total_width_mm)

baselines_per_row = compute_baseline_rows(H_T)
_, lead_x_ranges, _, _, _ = build_physical_template(H_T, W_T)

LAYOUT = {
    "lead_names": [
        "I", "II", "III",
        "aVR", "aVL", "aVF",
        "V1", "V2", "V3",
        "V4", "V5", "V6",
        "II_rhythm"
    ],
    "baseline_y": [
        baselines_per_row[1],  # I
        baselines_per_row[2],  # II
        baselines_per_row[3],  # III
        baselines_per_row[1],  # aVR
        baselines_per_row[2],  # aVL
        baselines_per_row[3],  # aVF
        baselines_per_row[1],  # V1
        baselines_per_row[2],  # V2
        baselines_per_row[3],  # V3
        baselines_per_row[1],  # V4
        baselines_per_row[2],  # V5
        baselines_per_row[3],  # V6
        baselines_per_row[4],  # II_rhythm
    ],
    "lead_x_ranges": [
        lead_x_ranges["I"],
        lead_x_ranges["II"],
        lead_x_ranges["III"],
        lead_x_ranges["aVR"],
        lead_x_ranges["aVL"],
        lead_x_ranges["aVF"],
        lead_x_ranges["V1"],
        lead_x_ranges["V2"],
        lead_x_ranges["V3"],
        lead_x_ranges["V4"],
        lead_x_ranges["V5"],
        lead_x_ranges["V6"],
        lead_x_ranges["II_rhythm"],
    ],  # not used in this code
}
# Slice truth according to lead
TRUTH_SLICING_FACTORS = {
    "I": (0.0, 0.25),
    "II": (0.0, 0.25),
    "III": (0.0, 0.25),
    "aVR": (0.25, 0.5),
    "aVL": (0.25, 0.5),
    "aVF": (0.25, 0.5),
    "V1": (0.5, 0.75),
    "V2": (0.5, 0.75),
    "V3": (0.5, 0.75),
    "V4": (0.75, 1.0),
    "V5": (0.75, 1.0),
    "V6": (0.75, 1.0),
    "II_rhythm": (0.0, 1.0),
}
