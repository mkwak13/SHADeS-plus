import pandas as pd
import os

# List of CSV file paths and corresponding method names
csv_files = [
    ("/raid/rema/outputs_rebuttal/undisttrain/undist/IID/finetuned_mono_hkfull_288/models/weights_19specscore_hkresults_2025-01-02.csv", "IID"),
    ("/raid/rema/outputs_rebuttal/undisttrain/undist/IID/finetuned_mono_hkfull_288/models/weights_19specscore_hkinpaintedresults_2025-01-02.csv", "IID_inp"),
    ("/raid/rema/outputs_rebuttal/undisttrain/undist/IID/finetuned_mono_hkfull_288_automasking/models/weights_19specscore_hkresults_2025-01-02.csv", "IID_am"),
    ("/raid/rema/outputs_rebuttal/undisttrain/undist/IID/finetuned_mono_hkfull_288_pseudo_dsms/models/weights_19specscore_hkresults_2025-01-02.csv", "IID_ps"),
    ("/raid/rema/outputs_rebuttal/undisttrain/undist/IID/finetuned_mono_hkfull_288_pseudo_dsms_automasking/models/weights_19specscore_hkresults_2025-01-02.csv", "IID_ps_am"),
    ("/raid/rema/outputs_rebuttal/undisttrain/undist/IID/finetuned_mono_hkfull_288_pseudo_dsms_automasking_noadjust/models/weights_19specscore_hkresults_2025-01-02.csv", "IID_ps_am_na"),
    ("/raid/rema/outputs_rebuttal/undisttrain/undist/IID/finetuned_mono_hkfull_288_pseudo_dsms_automasking_sploss/models/weights_19specscore_hkresults_2025-01-02.csv", "IID_ps_am_sploss"),
    ("/raid/rema/outputs_rebuttal/undisttrain/undist/monodepth2/finetuned_mono_hkfull_288/models/weights_19specscore_hkresults_2025-01-02.csv", "monodepth2"),
    ("/raid/rema/outputs_rebuttal/undisttrain/undist/monovit/finetuned_mono_hkfull_288/models/weights_19specscore_hkresults_2025-01-02.csv", "monovit")
    # Add more CSV files and method names as needed
]

# Initialize an empty DataFrame to store the combined results
combined_df = pd.DataFrame()

# Read each CSV file and merge the results
for csv_file, method_name in csv_files:
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Rename the mean_score column to the method name
    df = df.rename(columns={"SSM": method_name}).iloc[:, :-1]

    # Remove everything before and including "Inpainted_gen9/" or "Frames/" from the "path" column
    df["path"] = df["path"].str.replace(r'.*Inpainted_gen9/|.*Frames/', '', regex=True)
    
    # If combined_df is empty, initialize it with the current DataFrame
    if combined_df.empty:
        combined_df = df[["video", "path", method_name]]
    else:
        # Merge the current DataFrame with the combined DataFrame on the video column
        combined_df = pd.merge(combined_df, df[["video", "path", method_name]], on=["video", "path"], how="outer")

# Save the combined DataFrame to a new CSV file
combined_df.to_csv("/raid/rema/outputs_rebuttal/undisttrain/undist/combined_score_results.csv", index=False)

# get mean of every column and save
mean_df = combined_df.mean().reset_index()
mean_df.columns = ["method", "SSM"]

mean_df.T.to_csv("/raid/rema/outputs_rebuttal/undisttrain/undist/combined_score_results_mean.csv")

print("Combined results saved to combined_results.csv")