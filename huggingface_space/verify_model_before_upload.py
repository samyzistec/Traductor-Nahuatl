"""
Script para verificar que el modelo está bien configurado antes de subirlo
Ejecutar en Google Colab ANTES de subir a Hugging Face
"""

import json
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

# Ruta al modelo en Drive (ajustar si es diferente)
MODEL_PATH = "/content/drive/MyDrive/nllb-ncx-es-v3-FINAL"

print("=" * 80)
print("VERIFICACION DE MODELO NLLB")
print("=" * 80)

model_dir = Path(MODEL_PATH)

# 1. Verificar que la carpeta existe
if not model_dir.exists():
    print(f"ERROR: No se encontro el modelo en {MODEL_PATH}")
    print("Monta Google Drive y verifica la ruta.")
    exit(1)

print(f"OK: Carpeta encontrada: {MODEL_PATH}")

# 2. Verificar archivos necesarios
required_files = [
    "config.json",
    "generation_config.json",
    "pytorch_model.bin",  # o "model.safetensors"
    "tokenizer_config.json",
    "sentencepiece.bpe.model",
]

print("\nVerificando archivos:")
missing = []
for file in required_files:
    if (model_dir / file).exists():
        print(f"  OK: {file}")
    else:
        print(f"  FALTA: {file}")
        missing.append(file)

if "pytorch_model.bin" not in [f for f in (model_dir).glob("*")] and \
   "model.safetensors" not in [f for f in (model_dir).glob("*")]:
    print("  ADVERTENCIA: No se encontro pytorch_model.bin ni model.safetensors")
    print("  Busca archivos model-*.safetensors o pytorch_model-*.bin")

# 3. Verificar config.json
print("\nVerificando config.json:")
config_file = model_dir / "config.json"
if config_file.exists():
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    model_type = config.get("model_type", "DESCONOCIDO")
    print(f"  model_type: {model_type}")
    
    if model_type in ["m2m_100", "nllb", "nllb_moe"]:
        print(f"  OK: Tipo de modelo correcto para traduccion")
    else:
        print(f"  ERROR: model_type '{model_type}' NO es valido para NLLB")
        print(f"  Deberia ser 'm2m_100' o 'nllb'")
        print(f"  Tu modelo probablemente tiene el config.json incorrecto.")
        exit(1)
else:
    print("  ERROR: config.json no encontrado")
    exit(1)

# 4. Intentar cargar el modelo
print("\nIntentando cargar el modelo:")
try:
    print("  Cargando tokenizer...")
    tokenizer = NllbTokenizer.from_pretrained(MODEL_PATH)
    print(f"  OK: Tokenizer cargado (vocab: {len(tokenizer)} tokens)")
    
    print("  Cargando modelo (puede tardar 1-2 minutos)...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    print(f"  OK: Modelo cargado ({model.num_parameters():,} parametros)")
    
    # Prueba rápida
    print("\n  Haciendo prueba rapida...")
    tokenizer.src_lang = "nah_Latn"
    inputs = tokenizer("Cualli tonalli", return_tensors="pt")
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("spa_Latn"),
        max_length=50
    )
    traduccion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  Prueba: 'Cualli tonalli' -> '{traduccion}'")
    
    print("\n" + "=" * 80)
    print("TODO OK: El modelo esta listo para subirse a Hugging Face")
    print("=" * 80)
    
except Exception as e:
    print(f"\n  ERROR al cargar el modelo:")
    print(f"  {type(e).__name__}: {e}")
    print("\n  El modelo NO esta listo para subirse.")
    exit(1)


