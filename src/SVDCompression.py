import os
import gc
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD


# ==================================================
# CONFIG
# ==================================================
DATA_DIR = "../data/processed"

FILES = {
    "expert": {
        "input": "expert_features_with_embeddings.csv",
        "output": "expert_features_with_svd.csv",
        "prefix": "expert_emb_",
        "svd_prefix": "expert_svd_"
    },
    "user": {
        "input": "user_features_with_embeddings.csv",
        "output": "user_features_with_svd.csv",
        "prefix": "user_emb_",
        "svd_prefix": "user_svd_"
    }
}

N_COMPONENTS = 50
RANDOM_STATE = 42


# ==================================================
# SVD FUNCTION
# ==================================================
def run_svd(df, emb_prefix, svd_prefix, n_components):

    emb_cols = [c for c in df.columns if c.startswith(emb_prefix)]
    meta_cols = [c for c in df.columns if c not in emb_cols]

    print(f"Embedding columns found: {len(emb_cols)}")

    X = df[emb_cols].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    del X
    gc.collect()

    # SVD
    svd = TruncatedSVD(
        n_components=n_components,
        random_state=RANDOM_STATE
    )

    X_svd = svd.fit_transform(X_scaled)

    explained = svd.explained_variance_ratio_.sum()
    print(f"Explained variance retained: {explained:.2%}")

    del X_scaled
    gc.collect()

    svd_cols = [f"{svd_prefix}{i}" for i in range(n_components)]
    df_svd = pd.DataFrame(X_svd, columns=svd_cols)

    df_final = pd.concat(
        [
            df[meta_cols].reset_index(drop=True),
            df_svd.reset_index(drop=True)
        ],
        axis=1
    )

    return df_final


# ==================================================
# MAIN
# ==================================================
def main():

    print("Starting SVD compression for EXPERT and USER embeddings")
    print(f"Number of components: {N_COMPONENTS}")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    for name, cfg in FILES.items():
        print(f"\nProcessing {name.upper()} embeddings")

        input_path = os.path.join(DATA_DIR, cfg["input"])
        output_path = os.path.join(DATA_DIR, cfg["output"])

        df = pd.read_csv(input_path)
        print("Loaded shape:", df.shape)

        df_svd = run_svd(
            df=df,
            emb_prefix=cfg["prefix"],
            svd_prefix=cfg["svd_prefix"],
            n_components=N_COMPONENTS
        )

        df_svd.to_csv(output_path, index=False)
        print("Saved:", output_path)
        print("Final shape:", df_svd.shape)
        print("-" * 60)

    print("\n✔ SVD compression COMPLETE for expert & user")


if __name__ == "__main__":
    main()
