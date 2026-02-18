import pandas as pd

# Load Excel file and target sheet
file_path = "cpg_mtl_evaluation.xlsx"
sheet_name = "ablation-single-agent"
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Clean column names
df.columns = df.columns.str.strip()

# Target evaluation columns
target_columns = ["Syntactic Validity", "Semantic Accuracy", "Cyber-physical Consistency"]

# Fill NA as 1 (default correct), then cast to int
df[target_columns] = df[target_columns].fillna(1).astype(int)

# Total samples
total = len(df)

# Step 1: Syntactic Validity
df_syn = df[df["Syntactic Validity"] == 1]
syn_correct = len(df_syn)

# Step 2: Semantic Accuracy within syntactically valid
df_sem = df_syn[df_syn["Semantic Accuracy"] == 1]
sem_correct = len(df_sem)

# Step 3: Cyber-physical Consistency within previous two
df_cyber = df_sem[df_sem["Cyber-physical Consistency"] == 1]
cyber_correct = len(df_cyber)

# Overall correct: passed all three layers (you can also compute it directly)
overall_correct = len(df[
    (df["Syntactic Validity"] == 1) &
    (df["Semantic Accuracy"] == 1) &
    (df["Cyber-physical Consistency"] == 1)
])

# Results
results = {
    "Syntactic Validity": f"{syn_correct}/{total} ({syn_correct / total * 100:.1f}%)",
    "Semantic Accuracy": f"{sem_correct}/{syn_correct} ({sem_correct / syn_correct * 100:.1f}%)",
    "Cyber-physical Consistency": f"{cyber_correct}/{sem_correct} ({cyber_correct / sem_correct * 100:.1f}%)",
    "Overall Correctness": f"{overall_correct}/{total} ({overall_correct / total * 100:.1f}%)"
}

# Print results
for k, v in results.items():
    print(f"{k}: {v}")
