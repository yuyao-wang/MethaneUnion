import pandas as pd

base_dir = "/mnt/engg-leung/Research_No9_Methane_Emissions/Yuyao/train_tryout_32/carbonmapper_data_temporal_split_classification"
train_path = f"{base_dir}/train.csv"
test_path = f"{base_dir}/test.csv"


def balance_1_1(df, label_col="label", seed=42):
    pos = df[df[label_col] == 1]
    neg = df[df[label_col] == 0]

    # Translated comment
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError(f"One class is empty: pos={len(pos)}, neg={len(neg)}")

    # Translated comment
    if len(neg) > len(pos):
        neg = neg.sample(n=len(pos), random_state=seed)
    elif len(pos) > len(neg):
        pos = pos.sample(n=len(neg), random_state=seed)
    # Translated comment

    out = (
        pd.concat([pos, neg], ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    return out


train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_bal = balance_1_1(train_df)
test_bal = balance_1_1(test_df)

train_bal.to_csv(f"{base_dir}/train_balanced.csv", index=False)
test_bal.to_csv(f"{base_dir}/test_balanced.csv", index=False)

print("train:", len(train_bal), "label1:", train_bal["label"].sum())
print("test:", len(test_bal), "label1:", test_bal["label"].sum())
