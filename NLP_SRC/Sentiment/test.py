import os
import sys

os.environ["HF_HOME"] = r"F:\huggingface_cache"

print("1")
import torch

print("2")
from transformers import AutoModelForSequenceClassification, AutoTokenizer

print("3")

FINAL_PATH = r"E:\Multilingual NER and Sentiment Analysis System\models\sentiment\final"

tokenizer = AutoTokenizer.from_pretrained(FINAL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(FINAL_PATH)
model.eval()
print("4 - model loaded!")

id2label = model.config.id2label

def predict(text):
    inputs  = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits  = model(**inputs).logits
    scores  = torch.softmax(logits, dim=-1)[0]
    label   = id2label[scores.argmax().item()]
    score   = scores.max().item()
    return label, round(score, 4)

test_sentences = [
    # English
    ("English", "This product is absolutely amazing! Best purchase I've ever made."),
    ("English", "Terrible service, I am extremely disappointed and will never return."),
    ("English", "The package arrived on time, nothing special about it."),
    # French
    ("French",  "Ce produit est fantastique, je suis très satisfait!"),
    ("French",  "Service horrible, je ne recommande pas du tout."),
    # Arabic
    ("Arabic",  "هذا المنتج رائع جداً، أنصح به بشدة"),
    ("Arabic",  "تجربة سيئة جداً، لن أشتري منهم مرة أخرى"),
    # German
    ("German",  "Das Produkt ist absolut fantastisch, sehr zufrieden!"),
]


print("\nInference test:")
print("-" * 55)
for lang, text in test_sentences:
    label, score = predict(text)
    emoji = "🟢" if label == "positive" else "🔴" if label == "negative" else "🟡"
    print(f"{emoji} [{lang}] {label} ({score})  {text[:50]}...")