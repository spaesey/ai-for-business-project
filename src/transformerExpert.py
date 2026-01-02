import os
import gc
import torch
import pandas as pd
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModel


def clean_text(series):
    return (
        series.astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip(" '\"")
    )


def main():

    # -------------------------------
    # Environment
    # -------------------------------
    device = 0 if torch.cuda.is_available() else -1
    print("Using GPU" if device == 0 else "Using CPU")

    # -------------------------------
    # Load data
    # -------------------------------
    df = pd.read_csv(
        "../Metacritic dataset/ExpertReviews.csv",
        engine="python",
        on_bad_lines="skip"
    )

    df["Rev"] = clean_text(df["Rev"])
    print(f"Expert reviews: {len(df)}")

    # -------------------------------
    # Sentiment
    # -------------------------------
    sentiment = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device,
        truncation=True,
        max_length=512,
        torch_dtype=torch.float16 if device == 0 else torch.float32
    )

    batch_size = 128 if device == 0 else 64
    outputs = []

    for i in tqdm(range(0, len(df), batch_size), desc="Expert sentiment"):
        outputs.extend(sentiment(df["Rev"].iloc[i:i + batch_size].tolist()))

    df["sentiment_label"] = [o["label"] for o in outputs]
    df["sentiment_score"] = [
        o["score"] if o["label"] == "POSITIVE" else -o["score"]
        for o in outputs
    ]

    del sentiment, outputs
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------
    # Embeddings
    # -------------------------------
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained(
        "distilbert-base-uncased"
    ).to("cuda" if device == 0 else "cpu")
    model.eval()

    emb_batch = 64 if device == 0 else 32
    all_embeddings = []

    for i in tqdm(range(0, len(df), emb_batch), desc="Expert embeddings"):
        batch = df["Rev"].iloc[i:i + emb_batch].tolist()

        with torch.no_grad():
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(model.device)

            emb = model(**encoded).last_hidden_state.mean(dim=1)
            all_embeddings.append(emb.cpu())

        torch.cuda.empty_cache()

    embeddings = torch.cat(all_embeddings).numpy()
    emb_dim = embeddings.shape[1]

    emb_cols = [f"expert_emb_{i}" for i in range(emb_dim)]
    df = pd.concat([df, pd.DataFrame(embeddings, columns=emb_cols)], axis=1)

    del model, tokenizer, all_embeddings
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------
    # Aggregate
    # -------------------------------
    features = (
        df.groupby("url")
        .agg(
            expert_sentiment_mean=("sentiment_score", "mean"),
            expert_positive_ratio=("sentiment_label", lambda x: (x == "POSITIVE").mean()),
            expert_review_count=("sentiment_score", "count"),
            **{c: (c, "mean") for c in emb_cols}
        )
        .reset_index()
    )

    os.makedirs("../data/processed", exist_ok=True)
    features.to_csv(
        "../data/processed/expert_features_with_embeddings.csv",
        index=False
    )

    print("✔ Expert pipeline complete")


if __name__ == "__main__":
    main()
