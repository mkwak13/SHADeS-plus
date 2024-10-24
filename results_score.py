import pandas as pd
import os

# List of CSV file paths and corresponding method names
csv_files = [
    ("/raid/rema/outputs/undisttrain/undist/IID/finetuned_mono_hkfull_288/models/weights_19specscore_hkresults_2024-10-21.csv", "IID"),
    ("/raid/rema/outputs/undisttrain/undist/IID/finetuned_mono_hkfull_288/models/weights_19specscore_hkinpaintedresults_2024-10-21.csv", "IID_inp"),
    ("/raid/rema/outputs/undisttrain/undist/IID/finetuned_mono_hkfull_288_automasking/models/weights_19specscore_results_2024-10-21.csv", "IID_am"),
    ("/raid/rema/outputs/undisttrain/undist/IID/finetuned_mono_hkfull_288_pseudo_dsms/models/weights_19specscore_results_2024-10-21.csv", "IID_ps"),
    ("/raid/rema/outputs/undisttrain/undist/IID/finetuned_mono_hkfull_288_pseudo_dsms_automasking/models/weights_19specscore_results_2024-10-21.csv", "IID_ps_am"),
    ("/raid/rema/outputs/undisttrain/undist/IID/finetuned_mono_hkfull_288_pseudo_dsms_automasking_noadjust/models/weights_19specscore_results_2024-10-21.csv", "IID_ps_am_na"),
    ("/raid/rema/outputs/undisttrain/undist/IID/finetuned_mono_hkfull_288_pseudo_dsms_automasking_sploss/models/weights_19specscore_results_2024-10-21.csv", "IID_ps_am_sploss"),
    ("/raid/rema/outputs/undisttrain/undist/monodepth2/finetuned_mono_hkfull_288/models/weights_19specscore_results_2024-10-21.csv", "monodepth2"),
    ("/raid/rema/outputs/undisttrain/undist/monovit/finetuned_mono_hkfull_288/models/weights_19specscore_results_2024-10-21.csv", "monovit")
    # Add more CSV files and method names as needed
]

# Initialize an empty DataFrame to store the combined results
combined_df = pd.DataFrame()

# Read each CSV file and merge the results
for csv_file, method_name in csv_files:
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Rename the mean_score column to the method name
    df = df.rename(columns={"mean_score": method_name})
    
    # If combined_df is empty, initialize it with the current DataFrame
    if combined_df.empty:
        combined_df = df[["video", method_name]]
    else:
        # Merge the current DataFrame with the combined DataFrame on the video column
        combined_df = pd.merge(combined_df, df[["video", method_name]], on="video", how="outer")

# Save the combined DataFrame to a new CSV file
combined_df.to_csv("/raid/rema/outputs/undisttrain/undist/combined_score_results.csv", index=False)

print("Combined results saved to combined_results.csv")