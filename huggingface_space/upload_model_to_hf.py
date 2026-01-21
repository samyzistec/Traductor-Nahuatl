"""
Script para subir el modelo v3 a Hugging Face Hub desde Google Colab

Uso:
1. Ejecutar en una celda de Colab
2. Pegar tu token de HF cuando te lo pida
3. Esperar 20-30 minutos (el modelo es 5.2 GB)
"""

import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path
from huggingface_hub import HfApi, notebook_login, create_repo
import json

# ============================================================================
# CONFIGURACIÓN - EDITA ESTOS VALORES ANTES DE EJECUTAR
# ============================================================================
# IMPORTANTE: Cambia "samyzistec" por tu usuario de Hugging Face
REPO_ID = "samyzistec/nllb-ncx-es-v3"  # ← CAMBIAR: tu-usuario/tu-modelo
MODEL_PATH = "/content/drive/MyDrive/nllb-ncx-es-v3-FINAL"  # Ajustar si es diferente
REPO_TYPE = "model"

print("=" * 80)
print(" SUBIENDO MODELO A HUGGING FACE HUB")
print("=" * 80)

# Verificar que el modelo existe
model_dir = Path(MODEL_PATH)
if not model_dir.exists():
    print(f" ERROR: No se encontró el modelo en {MODEL_PATH}")
    print(f"\nVerifica que:")
    print(f"  1. Google Drive está montado")
    print(f"  2. La ruta del modelo es correcta")
    sys.exit(1)

# Listar archivos del modelo
print(f"\n Archivos encontrados en {MODEL_PATH}:")
model_files = list(model_dir.glob("*"))
total_size_mb = sum(f.stat().st_size for f in model_files if f.is_file()) / (1024**2)
print(f"  Total de archivos: {len(model_files)}")
print(f"  Tamaño total: {total_size_mb:.1f} MB")

required_files = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "sentencepiece.bpe.model",
]

print(f"\n✓ Verificando archivos requeridos:")
for file in required_files:
    if (model_dir / file).exists():
        print(f"   {file}")
    else:
        print(f"   {file} (FALTA)")

# Login
print(f"\n Iniciando login en Hugging Face...")
print(f"  → Pega tu token cuando te lo pida")
print(f"  → Crea un token en: https://huggingface.co/settings/tokens")
print(f"  → Necesita permisos de 'write'\n")

try:
    notebook_login()
    print(" Login exitoso")
except Exception as e:
    print(f" Error en login: {e}")
    sys.exit(1)

# Crear repositorio
print(f"\n Creando repositorio: {REPO_ID}...")
try:
    api = HfApi()
    create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, exist_ok=True, private=False)
    print(f" Repositorio creado: https://huggingface.co/{REPO_ID}")
except Exception as e:
    print(f" Advertencia al crear repo: {e}")
    print(f"  (Probablemente ya existe, continuando...)")

# Subir archivos
print(f"\n Subiendo modelo...")
print(f"  Esto puede tardar 20-30 minutos dependiendo de tu conexión")
print(f"  Tamaño total: {total_size_mb:.1f} MB\n")

try:
    api.upload_folder(
        folder_path=MODEL_PATH,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="Upload NLLB-ncx-es v3 (1.3B model, 18k corpus)",
    )
    print("\n ¡Modelo subido exitosamente!")
except Exception as e:
    print(f"\n Error al subir: {e}")
    sys.exit(1)

# Crear model card básico
print(f"\n Creando model card...")

model_card_content = """---
language:
- nah
- es
license: cc-by-nc-4.0
tags:
- translation
- náhuatl
- español
- nllb
- seq2seq
metrics:
- bleu
library_name: transformers
pipeline_tag: translation
---

# NLLB Náhuatl → Español (v3)

Modelo de traducción Náhuatl Central → Español basado en NLLB-200-distilled-1.3B.

## Uso

```python
from transformers import pipeline

translator = pipeline(
    "translation",
    model="samyzistec/nllb-ncx-es-v3",
    src_lang="nah_Latn",
    tgt_lang="spa_Latn"
)

result = translator("Cualli tonalli")
print(result[0]['translation_text'])
```

## Características

- **Corpus:** 18,173 pares náhuatl-español
- **BLEU:** 22.74
- **Modelo base:** facebook/nllb-200-distilled-1.3B
- **Dirección:** Náhuatl → Español (unidireccional)

Más información en: https://huggingface.co/spaces/samyzistec/traductor-nahuatl-v3
"""

try:
    api.upload_file(
        path_or_fileobj=model_card_content.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="Add model card",
    )
    print(" Model card creado")
except Exception as e:
    print(f" No se pudo crear model card: {e}")

# Resumen final
print("\n" + "=" * 80)
print(" ¡PROCESO COMPLETADO!")
print("=" * 80)
print(f"\n Enlaces:")
print(f"  Modelo: https://huggingface.co/{REPO_ID}")
print(f"  Prueba: https://huggingface.co/{REPO_ID}?text=Cualli+tonalli")
print(f"\n Próximos pasos:")
print(f"  1. Revisar el modelo en HF Hub")
print(f"  2. Editar el README.md si es necesario")
print(f"  3. Crear el Space para la demo")
print(f"  4. Compartir el link ")
print("\n" + "=" * 80)
