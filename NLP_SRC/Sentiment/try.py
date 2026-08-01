import os

os.environ["HF_HOME"] = r"F:\huggingface_cache"

from transformers import pipeline

FINAL_PATH = r"E:\Multilingual NER and Sentiment Analysis System\models\sentiment\final"

sent_pipe = pipeline(
    "text-classification",
    model=FINAL_PATH,
    tokenizer=FINAL_PATH,
)

test_sentences = [
    ("English", "This product is absolutely amazing! Best purchase I've ever made."),
    ("English", "Terrible service, I am extremely disappointed and will never return."),
    ("English", "The package arrived on time, nothing special about it."),
    ("French",  "Ce produit est fantastique, je suis très satisfait!"),
    ("French",  "Service horrible, je ne recommande pas du tout."),
    ("Arabic",  "هذا المنتج رائع جداً، أنصح به بشدة"),
    ("Arabic",  "تجربة سيئة جداً، لن أشتري منهم مرة أخرى"),
    ("German",  "Das Produkt ist absolut fantastisch, sehr zufrieden!"),
]

print("\nInference test:")
print("-" * 55)
for lang, text in test_sentences:
    r     = sent_pipe(text, truncation=True, max_length=128)[0]
    emoji = "🟢" if r["label"] == "positive" else "🔴" if r["label"] == "negative" else "🟡"
    print(f"{emoji} [{lang}] {r['label']} ({r['score']:.2f})  {text[:50]}...")