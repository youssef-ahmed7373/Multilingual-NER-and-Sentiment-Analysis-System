
import gradio as gr
import requests

API_URL = "http://localhost:8000"   # change if deployed remotely

ENTITY_COLORS = {
    "PER": {"bg": "#CECBF6", "text": "#3C3489"},
    "ORG": {"bg": "#9FE1CB", "text": "#085041"},
    "LOC": {"bg": "#FAC775", "text": "#633806"},
}

EXAMPLES = [
    ["Apple CEO Tim Cook announced a new partnership with Samsung in Seoul last Tuesday."],
    ["Emmanuel Macron met Elon Musk in Paris to discuss Tesla's European expansion."],
    ["هذا المنتج من شركة سامسونج كان مخيباً للآمال تماماً. اشتريته من أمازون وكانت التجربة سيئة."],
    ["Das Produkt von Volkswagen in Berlin war absolut fantastisch. Ich bin sehr zufrieden!"],
    ["Le président Emmanuel Macron a rencontré des représentants d'Airbus à Toulouse."],
]

# ── Helpers ──────────────────────────────────────────────────

def build_ner_html(text: str, entities: list) -> str:
    """Wrap entity spans in colored highlight tags."""
    if not entities:
        return f'<p style="font-size:15px;line-height:2">{text}</p>'

    sorted_ents = sorted(entities, key=lambda e: e["start"])
    parts, cursor = [], 0

    for ent in sorted_ents:
        parts.append(text[cursor:ent["start"]])
        colors = ENTITY_COLORS.get(ent["label"], {"bg": "#F1EFE8", "text": "#444"})
        parts.append(
            f'<mark style="background:{colors["bg"]};color:{colors["text"]};'
            f'padding:2px 7px;border-radius:4px;font-weight:500;margin:0 2px;">'
            f'{ent["text"]}'
            f'<sup style="font-size:10px;margin-left:3px;opacity:.75">{ent["label"]}</sup>'
            f'</mark>'
        )
        cursor = ent["end"]

    parts.append(text[cursor:])
    body = "".join(parts)
    return f'<p style="font-size:15px;line-height:2.2">{body}</p>'


def build_sentiment_html(sentiment: dict, language: str, ms: int) -> str:
    """Render the sentiment bar and metadata."""
    label    = sentiment["label"]
    score    = sentiment["score"]
    pct      = int(score * 100)
    is_pos   = label == "positive"
    is_neg   = label == "negative"
    color    = "#1D9E75" if is_pos else "#D85A30" if is_neg else "#EF9F27"
    emoji    = "🟢" if is_pos else "🔴" if is_neg else "🟡"

    return f"""
    <div style="font-family:sans-serif;padding:4px 0">
      <div style="font-size:12px;color:#888;margin-bottom:10px">
        Language: <strong>{language.upper()}</strong> &nbsp;·&nbsp; {ms}ms
      </div>
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
        <span style="font-size:22px">{emoji}</span>
        <span style="font-size:17px;font-weight:500;color:{color};text-transform:capitalize">{label}</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px">
        <div style="flex:1;height:10px;background:#eee;border-radius:5px;overflow:hidden">
          <div style="width:{pct}%;height:100%;background:{color};border-radius:5px"></div>
        </div>
        <span style="font-size:13px;color:#888;width:36px">{pct}%</span>
      </div>
    </div>"""


# ── Main prediction function ─────────────────────────────────

def predict(text: str):
    if not text.strip():
        return "", "", "⚠️ Please enter some text."

    try:
        resp = requests.post(
            f"{API_URL}/predict",
            json={"text": text},
            timeout=15,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        err = "❌ Cannot connect to API. Run: uvicorn api.main:app --port 8000"
        return "", "", err
    except requests.exceptions.HTTPError as e:
        return "", "", f"❌ API error: {e}"
    except Exception as e:
        return "", "", f"❌ Unexpected error: {e}"

    data     = resp.json()
    ner_html = build_ner_html(data["text"], data["entities"])
    sent_html = build_sentiment_html(data["sentiment"], data["language"], data["processing_ms"])

    entity_count = len(data["entities"])
    status = f"✅ Found {entity_count} entit{'y' if entity_count == 1 else 'ies'}"

    return ner_html, sent_html, status


# ── Gradio UI ────────────────────────────────────────────────

LEGEND_HTML = """
<div style="display:flex;gap:10px;flex-wrap:wrap;font-size:12px;margin-top:6px">
  <span style="background:#CECBF6;color:#3C3489;padding:2px 10px;border-radius:4px;font-weight:500">PER  person</span>
  <span style="background:#9FE1CB;color:#085041;padding:2px 10px;border-radius:4px;font-weight:500">ORG  organization</span>
  <span style="background:#FAC775;color:#633806;padding:2px 10px;border-radius:4px;font-weight:500">LOC  location</span>
</div>"""

with gr.Blocks(title="Multilingual NER + Sentiment") as demo:

    gr.Markdown("## 🌍 Multilingual NER & Sentiment Analysis")
    gr.Markdown(
        "Powered by **XLM-RoBERTa** fine-tuned on WikiANN (NER) and Cardiff Twitter Sentiment. "
        "Supports English, Arabic, French, German "
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input  = gr.Textbox(
                label="Input text",
                placeholder="Enter text in any language...",
                lines=5,
            )
            submit_btn  = gr.Button("Analyze", variant="primary", size="lg")
            status_text = gr.Textbox(label="", interactive=False, show_label=False)

        with gr.Column(scale=3):
            ner_output  = gr.HTML(label="Named entities")
            gr.HTML(LEGEND_HTML)
            gr.HTML("<hr style='margin:12px 0;border-color:var(--border-color-primary)'>")
            sent_output = gr.HTML(label="Sentiment")

    gr.Examples(
        examples=EXAMPLES,
        inputs=text_input,
        label="Try an example",
    )

    submit_btn.click(
        fn=predict,
        inputs=[text_input],
        outputs=[ner_output, sent_output, status_text],
    )

    # Also trigger on Enter key
    text_input.submit(
        fn=predict,
        inputs=[text_input],
        outputs=[ner_output, sent_output, status_text],
    )

if __name__ == "__main__":
    demo.launch(
        server_port=7860,
        share=True,    # set True to get a public link
    )
