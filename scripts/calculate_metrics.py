import pandas as pd
import numpy as np

ANIPOSE_CSV = "path/to/pose-3d.csv"
PREDICTIONS_CSV = "path/to/prediction_csv"

JOINTS = [
    "WRIST",

    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",

    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",

    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP", 

    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",

    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
]

anipose_df = pd.read_csv(ANIPOSE_CSV)
predictions_df = pd.read_csv(PREDICTIONS_CSV)

merged_df = predictions_df.merge(
    anipose_df,
    on='fnum',
    suffixes=("_pred", "_gt"),
)

print(f"Prediction frames: {len(predictions_df)}")
print(f"Anipose frames: {len(anipose_df)}")
print(f"Matched frames: {len(merged_df)}")

if len(merged_df) == 0:
    raise ValueError(
        "No matching frames were found"
        "Check the fnum column in both csv files"
    )

per_joint_results = {}

all_errors = []

for joint in JOINTS:

    pred_x = f"{joint}_x_pred"
    pred_y = f"{joint}_y_pred"
    pred_z = f"{joint}_z_pred"

    gt_x = f"{joint}_x_gt"
    gt_y = f"{joint}_y_gt"
    gt_z = f"{joint}_z_gt"

    required_columns = [
        pred_x,
        pred_y,
        pred_z,
        gt_x,
        gt_y,
        gt_z
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in merged_df.columns
    ]

    if missing_columns:
        print(
            f"Missing Columns for {joint}"
            f"{missing_columns}"
        )

        continue

    predicted = (
        merged_df[[pred_x, pred_y, pred_z]]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )

    ground_truth = (
            merged_df[[gt_x, gt_y, gt_z]]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float64)
        )

    errors = np.linalg.norm(
        predicted - ground_truth,
        axis=1
    )

    valid_errors = errors[~np.isnan(errors)]

    if len(valid_errors) == 0:
        print(f"No valid values for {joint}")
        continue

    mean_error = np.mean(valid_errors)

    per_joint_results[joint] = mean_error

    all_errors.extend(valid_errors)

mpjpe = np.mean(all_errors)

print("\n" + "=" * 60)
print("3d pose lifting results")
print("=" * 60)

print(f"\nMPJPE: {mpjpe:.4f}")

print("\nPer-Joint Error")
print("-" * 60)

for joint, error in per_joint_results.items():
    print(
        f"{joint:<25} "
        f"{error:.4f}"
    )

results = []

for joint, error in per_joint_results.items():

    results.append({
        "joint": joint,
        "mean_error": error
    })

results_df = pd.DataFrame(results)

results_df.to_csv("per_joint_error.csv", index=False)

summary_df = pd.DataFrame([
    {
        "metric": "MPJPE",
        "value": mpjpe
    }
])

summary_df.to_csv("evaluation_summary.csv", index=False)

print("\nResults saved")




