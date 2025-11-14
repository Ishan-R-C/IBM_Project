import os
from pathlib import Path
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import pandas as pd

st.title("Emotion Detection from Text")
st.write("Enter text and the model will predict emotions.")

def find_model_dir(base: Path) -> Path | None:
    if not base.exists():
        return None

    if any((base / f).exists() for f in ["config.json", "model.safetensors", "pytorch_model.bin"]):
        return base

    for child in base.iterdir():
        if child.is_dir():
            if any((child / f).exists() for f in ["config.json", "model.safetensors", "pytorch_model.bin"]):
                return child
    return None


@st.cache_resource(show_spinner=True)
def load_model():
    base = Path("roberta-emotion-model")
    model_dir = find_model_dir(base)

    if model_dir is None:
        alt = os.environ.get("ROBERTA_EMOTION_MODEL_PATH")
        if alt:
            model_dir = find_model_dir(Path(alt))

    if model_dir is None:
        raise FileNotFoundError("Model folder not found.")

    labels = ["anger", "fear", "joy", "neutral", "sadness", "surprise"]

    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        id2label={i: l for i, l in enumerate(labels)},
        label2id={l: i for i, l in enumerate(labels)},
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    return pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)


try:
    pipe = load_model()
except Exception as e:
    st.error(f"Model load error: {e}")
    pipe = None


text = st.text_area("Enter text:", "I am so happy today!")

if st.button("Analyze Emotion"):
    if pipe is None:
        st.error("Model unavailable.")
    else:
        with st.spinner("Analyzing..."):
            result = pipe(text)

        if isinstance(result, dict):
            result = [result]
        if isinstance(result, list) and len(result) and isinstance(result[0], list):
            result = result[0]

        df = pd.DataFrame(result)
        df["score"] = df["score"].astype(float)
        df = df.set_index("label")

        top = df["score"].idxmax()
        st.success(f"### Predicted Emotion: **{top.upper()}**")

        st.bar_chart(df["score"])

        
