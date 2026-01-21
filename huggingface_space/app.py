import os

import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, NllbTokenizer


MODEL_ID = os.getenv("HUB_MODEL_ID", "samyzistec/nllb-ncx-es-v3")
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "facebook/nllb-200-distilled-1.3B")
HF_TOKEN = os.getenv("HF_TOKEN")

SRC_NAH = "nah_Latn"
SRC_ESP = "spa_Latn"

device = "cuda" if torch.cuda.is_available() else "cpu"


def _load_tokenizer():
    """
    Intenta cargar el tokenizer desde el repo del modelo fine-tuned.
    Si no existe ahi (muy comun), hace fallback al tokenizer del modelo base.
    """
    last_err = None
    for candidate in [MODEL_ID, BASE_MODEL_ID]:
        try:
            tok = AutoTokenizer.from_pretrained(candidate, token=HF_TOKEN, use_fast=True)
            if not hasattr(tok, "src_lang"):
                tok = NllbTokenizer.from_pretrained(candidate, token=HF_TOKEN)
            return tok, candidate
        except Exception as e:
            last_err = e
    raise last_err  # pragma: no cover


tokenizer, TOKENIZER_SOURCE = _load_tokenizer()
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, token=HF_TOKEN).to(device)

print(f"Modelo cargado: {MODEL_ID} en {device}")
print(f"Tokenizer cargado desde: {TOKENIZER_SOURCE}")


def _translate(text: str, src_lang: str, tgt_lang: str, max_length: int) -> str:
    if not text or not text.strip():
        return ""

    tokenizer.src_lang = src_lang
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=int(max_length),
        num_beams=5,
        early_stopping=True,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


def translate_bidir(direction: str, text: str, max_length: int = 128):
    direction = (direction or "").strip()

    if direction == "Náhuatl -> Español":
        out = _translate(text, SRC_NAH, SRC_ESP, max_length)
        back = _translate(out, SRC_ESP, SRC_NAH, max_length) if out else ""
        return out, back

    out = _translate(text, SRC_ESP, SRC_NAH, max_length)
    back = _translate(out, SRC_NAH, SRC_ESP, max_length) if out else ""
    return out, back


def swap_direction_and_text(direction: str, inp: str, out: str):
    """
    Cambia la direccion y pone como nueva entrada la traduccion actual.
    """
    direction = (direction or "").strip()
    out = out or ""

    if direction == "Náhuatl -> Español":
        new_direction = "Español -> Náhuatl"
    else:
        new_direction = "Náhuatl -> Español"

    return new_direction, out


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""
# Traductor Náhuatl <-> Español

Modelo: `{MODEL_ID}`
Tokenizer: `{TOKENIZER_SOURCE}`

Selecciona la direccion. Al traducir, tambien veras la retraduccion (inversa) al mismo tiempo.
"""
    )

    with gr.Row():
        with gr.Column():
            direction = gr.Radio(
                choices=["Náhuatl -> Español", "Español -> Náhuatl"],
                value="Náhuatl -> Español",
                label="Direccion",
            )

            inp = gr.Textbox(label="Texto de entrada", lines=6)

            max_len = gr.Slider(
                16,
                256,
                value=128,
                step=1,
                label="Longitud maxima",
            )

            with gr.Row():
                btn = gr.Button("Traducir", variant="primary")
                swap_btn = gr.Button("Intercambiar")

        with gr.Column():
            out = gr.Textbox(label="Traduccion", lines=6)
            back = gr.Textbox(label="Traduccion inversa (re-traduccion)", lines=6)

    btn.click(translate_bidir, inputs=[direction, inp, max_len], outputs=[out, back])
    swap_btn.click(
        swap_direction_and_text,
        inputs=[direction, inp, out],
        outputs=[direction, inp],
    )

    gr.Examples(
        examples=[
            ["Náhuatl -> Español", "Cualli tonalli", 128],
            ["Náhuatl -> Español", "Nochi uan nikneki nitlatos", 128],
            ["Español -> Náhuatl", "Hola, ¿cómo estás?", 128],
            ["Español -> Náhuatl", "Buenos días", 128],
        ],
        inputs=[direction, inp, max_len],
        outputs=[out, back],
        fn=translate_bidir,
        cache_examples=False,
    )

    gr.Markdown(
        """
Notas:
- La calidad de Español -> Náhuatl puede variar segun el entrenamiento del modelo.
- Boton "Intercambiar": cambia la direccion y pone la traduccion como nueva entrada.
"""
    )


if __name__ == "__main__":
    demo.launch()


