# from transformers import pipeline
from transformers import pipeline

FINAL_PATH = r"E:\Multilingual NER and Sentiment Analysis System\models\ner\final"

def clean_entity(text):
    # Remove trailing punctuation
    text = text.strip(".,!?;:")
    # Add space before capital letters that follow lowercase (EmmanuelMacron → Emmanuel Macron)
    import re
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return text.strip()

ner_pipe = pipeline(
    "ner",
    model=FINAL_PATH,
    tokenizer=FINAL_PATH,
    aggregation_strategy="first",   # merges B-/I- spans automatically
)

test_sentences = [
    "Apple CEO Tim Cook announced a new partnership with Samsung in Seoul.",
    "Emmanuel Macron met Elon Musk in Paris to discuss Tesla.",
    "هذا المنتج من شركة سامسونج رائع جداً.",
]

print("\nInference test:")
for sentence in test_sentences:
    print(f"\nInput: {sentence}")
    entities = ner_pipe(sentence)
    for ent in entities:
        print(f"  [{ent['entity_group']}] {ent['word']}  (score: {ent['score']:.2f})")

print('--------------------------------------')
for sentence in test_sentences:
    print(f"\nInput: {sentence}")
    for ent in ner_pipe(sentence):
        word  = clean_entity(ent["word"])
        label = ent["entity_group"]
        score = ent["score"]
        print(f"  [{label}] {word}  ({score:.2f})")




