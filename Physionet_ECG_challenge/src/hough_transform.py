
import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_hough_lines(
    img_path,
    canny1=50,
    canny2=150,
    hough_threshold=120,
    min_line_length=80,
    max_line_gap=10,
    max_lines_to_draw=200,
):
    # 1) Read image (grayscale)
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    # 2) Edges
    edges = cv2.Canny(gray, canny1, canny2)

    # 3) Hough lines (segments)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    # Prepare color image for drawing
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    angles = []
    if lines is not None:
        # limit number of lines drawn
        lines_to_use = lines[:max_lines_to_draw]

        for (x1, y1, x2, y2) in lines_to_use[:, 0]:
            # draw line
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

            # compute angle in degrees
            ang = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
            angles.append(ang)

    print(f"Image: {img_path}")
    print("Edges nonzero pixels:", int(np.count_nonzero(edges)))
    if lines is None:
        print("HoughLinesP: no lines found (try lowering hough_threshold).")
        angles = np.array([])
    else:
        print("HoughLinesP: lines found:", len(lines))

        angles = np.array(angles, dtype=np.float32)
        # Normalize angles to [-90, +90] (so horizontals cluster around 0)
        angles = ((angles + 90) % 180) - 90

        print("Angle stats (deg) after normalization to [-90,90]:")
        print("  median:", float(np.median(angles)))
        print("  mean  :", float(np.mean(angles)))
        print("  std   :", float(np.std(angles)))

        # Print a few angles
        print("First 10 angles:", angles[:10].tolist())

    # 4) Display: original, edges, overlay
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 3, 1)
    plt.title("Grayscale")
    plt.imshow(gray, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Canny edges")
    plt.imshow(edges, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Hough overlay (red lines)")
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return lines, angles

ecg_id = "10140238"
base_path = "../data/sample"
img_path = f"{base_path}/{ecg_id}/{ecg_id}-0006.png"
ecg_metadata_path = "../data/train.csv"

lines, angles = show_hough_lines(img_path)
