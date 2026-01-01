# Transformer-Based Feature Extraction from Movie Reviews
#
# Optimized GPU version with clear progress output
# Safe for Windows + PyTorch + Hugging Face pipelines

import os
import math
import torch
import pandas as pd


def main():

    # ==============================================================
    print("\n=== Stage 0: Environment Check ===")

    device = 0 if torch.cuda.is_available() else -1
    print("Using GPU" if device == 0 else "Using CPU")
    print("CUDA available:", torch.cuda.is_available())

    from transformers import pipeline

    # ==============================================================
    print("\n=== Stage 1: Load Cleaned Review Data ===")

    expert_df = pd.read_csv("../cleaned_data/expert_reviews_clean.csv")
    user_df   = pd.read_csv("../cleaned_data/user_reviews_clean.csv")

    expert_df = expert_df.head(500)
    user_df = user_df.head(500)

    print(f"✔ Expert reviews: {len(expert_df)}")
    print(f"✔ User reviews:   {len(user_df)}")

    # ==============================================================
    print("\n=== Stage 2: Initialize Sentiment Transformer ===")

    sentiment_pipeline = pipeline(
        task="sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device,
        truncation=True,
        padding=True,
        torch_dtype=torch.float16 if device == 0 else torch.float32
    )

    print("✔ Sentiment pipeline initialized")
    print("✔ Model dtype:", sentiment_pipeline.model.dtype)

    # ==============================================================
    print("\n=== Stage 3: Prepare Text Data ===")

    expert_texts = expert_df["review_text"].fillna("").tolist()
    user_texts   = user_df["review_text"].fillna("").tolist()

    all_texts = expert_texts + user_texts
    split_idx = len(expert_texts)

    print(f"✔ Total texts to process: {len(all_texts)}")

    # ==============================================================
    print("\n=== Stage 4: Batched Sentiment Inference ===")

    batch_size = 100 if device == 0 else 64
    total = len(all_texts)
    num_batches = math.ceil(total / batch_size)

    print(f"✔ Batch size: {batch_size}")
    print(f"✔ Total batches: {num_batches}")
    print("✔ Running inference...")

    all_outputs = []

    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch_id = (i // batch_size) + 1

            # Minimal progress printing (fast)
            if batch_id == 1 or batch_id % 25 == 0 or batch_id == num_batches:
                print(f"  Batch {batch_id}/{num_batches}")

            batch = all_texts[i:i + batch_size]
            all_outputs.extend(sentiment_pipeline(batch))

    print("✔ Inference complete")

    # ==============================================================
    print("\n=== Stage 5: Split Results Back ===")

    expert_outputs = all_outputs[:split_idx]
    user_outputs   = all_outputs[split_idx:]

    # Expert
    expert_df["sentiment_label"] = [r["label"] for r in expert_outputs]
    expert_df["sentiment_score"] = [
        r["score"] if r["label"] == "POSITIVE" else -r["score"]
        for r in expert_outputs
    ]

    # User
    user_df["sentiment_label"] = [r["label"] for r in user_outputs]
    user_df["sentiment_score"] = [
        r["score"] if r["label"] == "POSITIVE" else -r["score"]
        for r in user_outputs
    ]

    print("✔ Sentiment columns added")

    # ==============================================================
    print("\n=== Stage 6: Aggregate at Movie Level ===")

    expert_movie_features = (
        expert_df.groupby("url")
        .agg(
            expert_sentiment_mean=("sentiment_score", "mean"),
            expert_sentiment_std=("sentiment_score", "std"),
            expert_positive_ratio=("sentiment_label", lambda x: (x == "POSITIVE").mean()),
            expert_review_count=("sentiment_score", "count")
        )
        .reset_index()
    )

    user_movie_features = (
        user_df.groupby("url")
        .agg(
            user_sentiment_mean=("sentiment_score", "mean"),
            user_sentiment_std=("sentiment_score", "std"),
            user_positive_ratio=("sentiment_label", lambda x: (x == "POSITIVE").mean()),
            user_review_count=("sentiment_score", "count")
        )
        .reset_index()
    )

    print(f"✔ Expert movies aggregated: {len(expert_movie_features)}")
    print(f"✔ User movies aggregated:   {len(user_movie_features)}")

    # ==============================================================
    print("\n=== Stage 7: Export Features ===")

    os.makedirs("../data/processed", exist_ok=True)

    expert_movie_features.to_parquet(
        "../data/processed/expert_sentiment_features.parquet",
        index=False
    )

    user_movie_features.to_parquet(
        "../data/processed/user_sentiment_features.parquet",
        index=False
    )

    print("✔ Features exported successfully")
    print("\n=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
