# Traductor Náhuatl-Español v4

Sistema de traducción automática Náhuatl Central (ncx) → Español basado en NLLB-200-distilled-1.3B.

## Especificaciones Técnicas

- **Modelo base:** `facebook/nllb-200-distilled-1.3B`
- **Arquitectura:** Seq2Seq Transformer (M2M100)
- **Parámetros:** 1.37B
- **Idiomas:** `nah_Latn` → `spa_Latn`
- **Corpus v4:** 18,173 pares únicos
- **BLEU Score:** 22.74
- **Framework:** Hugging Face Transformers 4.45.0+
- **Python:** 3.8+

## Estructura del Proyecto

```
Proyecto-Traductor/
├── data/
│   ├── raw/                          # Datos sin procesar
│   │   ├── jw_org/                   # Corpus bíblico JW.org
│   │   │   └── parallel_ncx_es.jsonl # 5,357 pares
│   │   ├── pdfs/                     # PDFs fuente para extracción
│   │   ├── axolotl/                  # Corpus Axolotl (descargado)
│   │   └── tatoeba/                  # Corpus Tatoeba (descargado)
│   ├── processed/                    # Corpus procesados (generado)
│   │   ├── corpus_v3.jsonl           # Corpus completo (18,173 pares)
│   │   ├── dictionary_puebla.jsonl   # Extraído de PDFs
│   │   ├── dictionary_hueyapan.jsonl # Extraído de PDFs
│   │   ├── inali_books.jsonl        # Extraído de PDFs
│   │   ├── nhc_parallels.jsonl      # Extraído de PDFs
│   │   └── parallel_ncx_es_EXPANDED.jsonl # Axolotl + Tatoeba
│   └── train_test_splits/            # Splits para entrenamiento (generado)
│       ├── train.jsonl               # 14,538 pares (80%)
│       ├── validation.jsonl          # 1,817 pares (10%)
│       └── test.jsonl                 # 1,818 pares (10%)
│
├── scripts/                          # Scripts de procesamiento
│   ├── extract_dictionary_puebla.py  # Extrae diccionario Puebla de PDF
│   ├── extract_dictionary_hueyapan.py # Extrae diccionario Hueyapan
│   ├── extract_inali_books.py       # Extrae libros INALI
│   ├── extract_nhc_parallels.py      # Extrae paralelos NHC
│   ├── download_corpus_FINAL.py     # Descarga Axolotl + Tatoeba
│   ├── build_corpus_v3.py            # Construye corpus completo
│   └── validate_corpus_v3.py        # Valida calidad del corpus
│
├── notebooks/                        # Notebooks de entrenamiento
│   ├── TRAIN_NLLB_v3_OPTIMIZED.ipynb # Entrenamiento optimizado (Colab)
│   └── TRAIN_NLLB_v3_COMPLETE.ipynb  # Entrenamiento completo (Colab)
│
├── huggingface_space/                # Deployment en Hugging Face
│   ├── app.py                        # Interfaz Gradio
│   ├── requirements.txt              # Dependencias del Space
│   └── upload_model_to_hf.py         # Script para subir modelo
│
├── src/                              # Código fuente modular
│   ├── extractors/                   # Extractores de datos
│   ├── preprocessing/                # Normalización y validación
│   ├── training/                     # Scripts de entrenamiento
│   ├── utils/                        # Utilidades
│   └── visualization/                # Visualización
│
├── legacy/                           # Código de versiones anteriores
└── requirements.txt                  # Dependencias Python
```

## Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip
- Git (para clonar el repositorio)
- ~5 GB espacio en disco
- Conexión a internet (para descargar corpus y modelo base)

### Instalación de Dependencias

```bash
# Clonar repositorio
git clone <repository-url>
cd Proyecto-Traductor

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### Verificación de Instalación

```bash
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
```

##  Guía Completa: Ejecutar en Colab y Desplegar en Hugging Face

Esta guía te llevará paso a paso desde el entrenamiento en Google Colab hasta el despliegue en Hugging Face.

---

###  PARTE 1: Preparación de Datos

#### Paso 1.1: Generar los Splits de Entrenamiento

Si aún no tienes los splits, ejecuta localmente:

```bash
# Construir corpus completo
python scripts/build_corpus_v3.py

# Esto genera:
# - data/train_test_splits/train.jsonl (14,538 pares)
# - data/train_test_splits/validation.jsonl (1,817 pares)
# - data/train_test_splits/test.jsonl (1,818 pares)
```

#### Paso 1.2: Subir Datos a Google Drive

1. Abre Google Drive: https://drive.google.com
2. Crea una carpeta llamada `nahuatl_corpus_v3` (o `nahuatl_corpus_v4` si prefieres)
3. Sube los 3 archivos JSONL:
   - `train.jsonl`
   - `validation.jsonl`
   - `test.jsonl`

**Ubicación final en Drive:** `MyDrive/nahuatl_corpus_v3/`

---

###  PARTE 2: Entrenamiento en Google Colab

#### Paso 2.1: Abrir Notebook en Colab

1. Ve a [Google Colab](https://colab.research.google.com/)
2. Sube el notebook: `notebooks/TRAIN_NLLB_v3_OPTIMIZED.ipynb`
   - O clona el repo directamente en Colab:
   ```python
   !git clone <tu-repo-url>
   %cd Proyecto-Traductor
   ```

#### Paso 2.2: Configurar GPU

1. En Colab: **Runtime → Change runtime type**
2. Configuración:
   - **Hardware accelerator:** GPU
   - **GPU type:** A100 (preferible) o T4
   - Click **Save**

#### Paso 2.3: Ejecutar el Notebook

El notebook está dividido en secciones numeradas. Ejecuta las celdas en orden:

**Sección 1-3: Verificación**
- Verifica GPU disponible
- Instala dependencias necesarias

**Sección 4-6: Configuración**
- Monta Google Drive (te pedirá autorización)
- Configura rutas de datos y salida
- **IMPORTANTE:** El modelo se guardará en `MyDrive/nllb-ncx-es-v3-FINAL/`

**Sección 7-10: Carga de Datos**
- Carga el corpus desde Drive
- Muestra estadísticas

**Sección 11-13: Carga de Modelo**
- Descarga el modelo base NLLB-1.3B (~2.6 GB)
- Carga tokenizer y modelo en GPU

**Sección 14-18: Preprocesamiento y Métricas**
- Tokeniza los datos
- Configura métrica BLEU

**Sección 19-21: Configuración de Entrenamiento**
- Define hiperparámetros
- Crea el trainer

**Sección 22-23:  ENTRENAMIENTO**
```python
train_result = trainer.train()
```
-  **Tiempo estimado:** 2.5-3 horas
-  **Guardado automático:** Cada 500 steps en Drive
-  **Evaluación:** Cada 500 steps
-  **Puedes cerrar la pestaña:** El modelo se guarda en Drive

**Sección 24-26: Guardado y Evaluación**
- Guarda modelo final
- Evalúa en test set
- Genera reportes

**Sección 27-30: Pruebas**
- Prueba traducciones de ejemplo

**Resultado Final:**
-  Modelo guardado en: `MyDrive/nllb-ncx-es-v3-FINAL/`
-  Archivos: `config.json`, `model.safetensors`, `tokenizer_config.json`, etc.

---

###  PARTE 3: Subir Modelo a Hugging Face Hub

#### Paso 3.1: Obtener Token de Acceso

1. Ve a: https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Configuración:
   - **Name:** `colab-upload` (o el nombre que prefieras)
   - **Type:** **Write** (necesario para subir)
4. Click **"Generate token"**
5. ** COPIA EL TOKEN INMEDIATAMENTE** (solo se muestra una vez)

#### Paso 3.2: Subir Modelo desde Colab

**Opción A: Usar Script Incluido (Recomendado)**

En una nueva celda de Colab, después del entrenamiento:

```python
# Subir script a Colab
!wget https://raw.githubusercontent.com/tu-usuario/tu-repo/main/huggingface_space/upload_model_to_hf.py
# O si ya está en el repo clonado:
# %cd Proyecto-Traductor

# Editar el REPO_ID en el script (línea 20)
# Cambiar "samyzistec/nllb-ncx-es-v3" por "tu-usuario/tu-modelo"

# Ejecutar
exec(open('huggingface_space/upload_model_to_hf.py').read())
```

Cuando te pida el token, pégalo.

**Opción B: Código Manual**

En una nueva celda de Colab:

```python
from huggingface_hub import login, HfApi, create_repo

# 1. Login (pega tu token cuando te lo pida)
login()

# 2. Configuración
REPO_ID = "tu-usuario/nllb-ncx-es-v3"  # Cambia por tu usuario
MODEL_PATH = "/content/drive/MyDrive/nllb-ncx-es-v3-FINAL"

# 3. Crear repositorio
api = HfApi()
create_repo(
    repo_id=REPO_ID,
    repo_type="model",
    exist_ok=True
)

# 4. Subir modelo (esto tarda 20-30 minutos)
print(" Subiendo modelo... Esto puede tardar 20-30 minutos")
api.upload_folder(
    folder_path=MODEL_PATH,
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Upload NLLB-ncx-es v3 (1.3B, 18k corpus)"
)

print(f" ¡Modelo subido! Visita: https://huggingface.co/{REPO_ID}")
```

** Tiempo estimado:** 20-30 minutos (modelo ~5 GB)

#### Paso 3.3: Crear Model Card

1. Ve a tu modelo en Hugging Face: `https://huggingface.co/tu-usuario/nllb-ncx-es-v3`
2. Click en **"Edit model card"** (botón de lápiz)
3. Copia y pega este contenido:

```markdown
---
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

# Traductor Náhuatl → Español (v3)

Modelo de traducción Náhuatl Central → Español basado en NLLB-200-distilled-1.3B.

## Uso

```python
from transformers import pipeline

translator = pipeline(
    "translation",
    model="tu-usuario/nllb-ncx-es-v3",
    src_lang="nah_Latn",
    tgt_lang="spa_Latn"
)

result = translator("Cualli tonalli")
print(result[0]['translation_text'])  # "Buenos días"
```

## Métricas

- **BLEU Score:** 22.74
- **Corpus:** 18,173 pares náhuatl-español
- **Modelo base:** facebook/nllb-200-distilled-1.3B
- **Parámetros:** 1.37B

## Corpus

| Fuente | Pares | Porcentaje |
|--------|-------|------------|
| Axolotl + Tatoeba | 9,156 | 50.3% |
| JW.org | 5,357 | 29.4% |
| Diccionarios | 3,554 | 19.5% |
| INALI | 143 | 0.8% |
| **TOTAL** | **18,173** | **100%** |
```

4. Click **"Save"**

---

###  PARTE 4: Crear Hugging Face Space (Demo Interactiva)

#### Paso 4.1: Crear el Space

1. Ve a: https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Configuración:
   - **Space name:** `traductor-nahuatl-v3` (o el nombre que prefieras)
   - **SDK:** **Gradio**
   - **Hardware:** **CPU basic** (gratis)
   - **Visibility:** **Public**
4. Click **"Create Space"**

#### Paso 4.2: Subir Archivos

El Space se crea con un repositorio Git. Tienes 3 opciones:

**Opción A: Subir desde la Web (Más Fácil)**

1. En la página del Space, click en **"Files and versions"**
2. Click en **"Add file" → "Upload files"**
3. Sube estos archivos desde `huggingface_space/`:
   - `app.py`
   - `requirements.txt`

**Opción B: Usar Git (Recomendado para actualizaciones)**

```bash
# Clonar el Space
git clone https://huggingface.co/spaces/tu-usuario/traductor-nahuatl-v3
cd traductor-nahuatl-v3

# Copiar archivos
cp /ruta/a/Proyecto-Traductor/huggingface_space/app.py .
cp /ruta/a/Proyecto-Traductor/huggingface_space/requirements.txt .

# Editar app.py: Cambiar MODEL_ID (línea 6) por tu modelo
# MODEL_ID = "tu-usuario/nllb-ncx-es-v3"

# Commit y push
git add app.py requirements.txt
git commit -m "Add Gradio app"
git push
```

**Opción C: Editar Directamente en la Web**

1. En el Space, click en **"Files and versions"**
2. Click en **"Add file" → "Create a new file"**
3. Crea `app.py` y copia el contenido de `huggingface_space/app.py`
4. **IMPORTANTE:** Edita la línea 6 para usar tu modelo:
   ```python
   MODEL_ID = os.getenv("HUB_MODEL_ID", "tu-usuario/nllb-ncx-es-v3")
   ```
5. Crea `requirements.txt` con:
   ```
   transformers==4.45.0
   torch==2.0.0
   gradio==4.44.0
   sentencepiece==0.1.99
   ```

#### Paso 4.3: Configurar Variable de Entorno (Opcional)

Si quieres que el Space use una variable de entorno para el modelo:

1. En el Space, ve a **"Settings"**
2. Scroll hasta **"Repository secrets"**
3. Agrega:
   - **Key:** `HUB_MODEL_ID`
   - **Value:** `tu-usuario/nllb-ncx-es-v3`
4. Click **"Add secret"**

#### Paso 4.4: Esperar el Build

1. El Space se construye automáticamente
2. Ve a la pestaña **"Logs"** para ver el progreso
3.  **Tiempo estimado:** 5-10 minutos
4. Cuando termine, verás **"Running"** en verde

#### Paso 4.5: Probar el Space

1. Ve a la pestaña **"App"**
2. Prueba con ejemplos:
   - `Cualli tonalli` → Debería traducir a "Buenos días"
   - `Ne Dios kuali tejuanti` → "Dios nos ama"

---

###  Resumen de Pasos

**Entrenamiento en Colab:**
1.  Subir datos a Google Drive
2.  Abrir notebook en Colab
3.  Configurar GPU
4.  Ejecutar todas las celdas
5.  Esperar 2.5-3 horas
6.  Modelo guardado en Drive

**Subir a Hugging Face:**
1.  Obtener token de acceso (Write)
2.  Ejecutar script de subida en Colab
3.  Esperar 20-30 minutos
4.  Crear model card

**Crear Space:**
1.  Crear nuevo Space (Gradio)
2.  Subir `app.py` y `requirements.txt`
3.  Editar MODEL_ID en `app.py`
4.  Esperar build (5-10 min)
5.  Probar la demo

---

###  Enlaces Útiles

- **Google Colab:** https://colab.research.google.com
- **Hugging Face Tokens:** https://huggingface.co/settings/tokens
- **Hugging Face Spaces:** https://huggingface.co/spaces
- **Documentación NLLB:** https://huggingface.co/docs/transformers/model_doc/nllb

---

## Pipeline de Construcción del Corpus

### Opción 1: Corpus Completo (Recomendado)

Requiere conexión a internet. Genera ~18,173 pares.

#### Paso 1: Extracción de PDFs

Los scripts de extracción procesan archivos Markdown generados previamente desde PDFs:

```bash
# Diccionario Náhuatl Norte de Puebla (~3,166 pares)
python scripts/extract_dictionary_puebla.py
# Genera: data/processed/dictionary_puebla.jsonl

# Diccionario Náhuatl de Hueyapan (~388 pares)
python scripts/extract_dictionary_hueyapan.py
# Genera: data/processed/dictionary_hueyapan.jsonl

# Libros INALI (~143 pares)
python scripts/extract_inali_books.py
# Genera: data/processed/inali_books.jsonl

# Paralelos NHC (~6 pares, opcional)
python scripts/extract_nhc_parallels.py
# Genera: data/processed/nhc_parallels.jsonl
```

**Nota:** Los scripts esperan archivos `.md` en `data/raw/pdfs/`. Si los PDFs no están convertidos, se requiere procesamiento previo con herramientas OCR.

#### Paso 2: Descarga de Corpus Adicionales

```bash
# Descarga Axolotl Corpus + Tatoeba (~9,156 pares)
python scripts/download_corpus_FINAL.py
# Genera: data/processed/parallel_ncx_es_EXPANDED.jsonl
```

**Funcionamiento:**
- Descarga corpus Axolotl desde repositorio GitHub
- Descarga corpus Tatoeba vía API
- Filtra contenido bíblico duplicado
- Combina y deduplica
- Guarda en formato JSONL

#### Paso 3: Construcción del Corpus Completo

```bash
# Combina todas las fuentes y genera splits
python scripts/build_corpus_v3.py
```

**Proceso interno:**
1. Carga todas las fuentes disponibles:
   - `data/raw/jw_org/parallel_ncx_es.jsonl` (5,357 pares)
   - `data/processed/parallel_ncx_es_EXPANDED.jsonl` (9,156 pares)
   - `data/processed/dictionary_puebla.jsonl` (3,166 pares)
   - `data/processed/dictionary_hueyapan.jsonl` (388 pares)
   - `data/processed/inali_books.jsonl` (143 pares)
   - `data/processed/nhc_parallels.jsonl` (6 pares)

2. Normalización:
   - Elimina espacios múltiples
   - Normaliza encoding UTF-8
   - Valida longitud mínima (2 caracteres)

3. Deduplicación:
   - Compara pares normalizados (case-insensitive)
   - Elimina duplicados exactos
   - Mantiene metadatos de la primera ocurrencia

4. Generación de splits:
   - Train: 80% (14,538 pares)
   - Validation: 10% (1,817 pares)
   - Test: 10% (1,818 pares)
   - Semilla aleatoria: 42 (reproducible)

**Salidas:**
- `data/processed/corpus_v3.jsonl` (18,173 pares)
- `data/train_test_splits/train.jsonl`
- `data/train_test_splits/validation.jsonl`
- `data/train_test_splits/test.jsonl`

#### Paso 4: Validación

```bash
python scripts/validate_corpus_v3.py
```

**Métricas generadas:**
- Estadísticas básicas (total, longitudes)
- Distribución por fuente y dominio
- Detección de problemas (pares muy cortos/largos, ratios extremos)
- Vocabulario único (Náhuatl y Español)
- Ejemplos aleatorios por dominio

### Opción 2: Corpus Parcial (Sin Internet)

Si no hay conexión a internet, se puede construir un corpus parcial usando solo datos locales:

```bash
# Solo extraer PDFs (requiere archivos .md en data/raw/pdfs/)
python scripts/extract_dictionary_puebla.py
python scripts/extract_dictionary_hueyapan.py
python scripts/extract_inali_books.py

# Construir corpus (usará solo JW.org + PDFs)
python scripts/build_corpus_v3.py
# Resultado: ~9,000 pares (sin Axolotl/Tatoeba)
```

## Entrenamiento del Modelo

### Configuración de Entrenamiento

**Hiperparámetros:**
- Modelo base: `facebook/nllb-200-distilled-1.3B`
- Epochs: 3
- Batch size efectivo: 16 (per_device=4, gradient_accumulation=4)
- Learning rate: 2e-5
- Optimizador: AdamW
- FP16: True
- Max length: 256 tokens
- Warmup steps: 500
- Eval strategy: steps (cada 250 steps)
- Save strategy: steps (cada 250 steps)

**Recursos requeridos:**
- GPU: A100 (40GB) o T4 (16GB)
- VRAM mínimo: 16GB
- Tiempo estimado: 2-3 horas (A100) o 4-6 horas (T4)

### Proceso de Entrenamiento

#### 1. Preparación de Datos en Google Drive

Subir a Google Drive en carpeta `MyDrive/nahuatl_corpus_v4/`:
- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`

#### 2. Configuración de Google Colab

1. Abrir [Google Colab](https://colab.research.google.com/)
2. Subir notebook: `notebooks/TRAIN_NLLB_v3_OPTIMIZED.ipynb`
3. Configurar GPU:
   - Runtime → Change runtime type
   - Hardware accelerator: GPU
   - GPU type: A100 (preferible) o T4

#### 3. Ejecución del Notebook

El notebook `TRAIN_NLLB_v3_OPTIMIZED.ipynb` contiene:

**Celdas 1-3: Verificación y Instalación**
- Verifica GPU disponible (`nvidia-smi`)
- Instala dependencias (transformers, datasets, accelerate, etc.)

**Celdas 4-6: Montar Drive y Configuración**
- Monta Google Drive
- Configura rutas de datos y salida
- Define constantes (MODEL_NAME, SRC_LANG, TGT_LANG, MAX_LENGTH)

**Celdas 7-10: Carga de Datos**
- Carga corpus desde Drive usando `load_dataset`
- Muestra estadísticas y ejemplos

**Celdas 11-13: Carga de Modelo**
- Descarga tokenizer NllbTokenizer
- Descarga modelo M2M100ForConditionalGeneration
- Mueve modelo a GPU

**Celdas 14-16: Preprocesamiento**
- Define función `preprocess_function`:
  - Tokeniza fuente (Náhuatl) con `src_lang="nah_Latn"`
  - Tokeniza target (Español) con `src_lang="spa_Latn"`
  - Aplica truncation y padding
- Aplica preprocesamiento a datasets
- Crea DataCollatorForSeq2Seq

**Celdas 17-18: Configuración de Métricas**
- Configura métrica BLEU (sacrebleu)
- Define función `compute_metrics` con manejo de errores

**Celdas 19-20: Configuración de Entrenamiento**
- Crea `Seq2SeqTrainingArguments`:
  - Output directory en Drive
  - Batch sizes y learning rate
  - Estrategias de evaluación y guardado
  - FP16 activado
- Crea `Seq2SeqTrainer` con datasets, modelo, tokenizer, collator y métricas

**Celda 21: Entrenamiento**
```python
train_result = trainer.train()
```
- Ejecuta fine-tuning
- Guarda checkpoints cada 250 steps en Drive
- Evalúa cada 250 steps
- Selecciona mejor modelo automáticamente

**Celdas 22-24: Guardado y Evaluación**
- Guarda modelo final en Drive
- Evalúa en test set
- Genera reporte de métricas

**Resultado:**
- Modelo guardado en: `MyDrive/nllb-ncx-es-v4-FINAL/`
- Archivos: `config.json`, `model.safetensors`, `tokenizer_config.json`, `sentencepiece.bpe.model`, etc.

## Deployment en Hugging Face

### Subir Modelo a Hugging Face Hub

#### 1. Obtener Token de Acceso

1. Ir a https://huggingface.co/settings/tokens
2. Crear nuevo token con permisos **Write**
3. Copiar token (solo se muestra una vez)

#### 2. Subir desde Google Colab

**Opción A: Usar script incluido**

En Colab, después del entrenamiento:

```python
# Ejecutar script de subida
exec(open('huggingface_space/upload_model_to_hf.py').read())
```

**Opción B: Código manual**

```python
from huggingface_hub import login, HfApi

# Login
login()  # Pega tu token cuando te lo pida

# Subir modelo
api = HfApi()
api.upload_folder(
    folder_path="/content/drive/MyDrive/nllb-ncx-es-v4-FINAL",
    repo_id="tu-usuario/nllb-ncx-es-v4",
    repo_type="model",
    commit_message="Upload NLLB-ncx-es v4 (1.3B, 18k corpus)"
)
```

**Tiempo estimado:** 20-30 minutos (modelo ~5GB)

#### 3. Crear Model Card

Editar README.md del modelo en Hugging Face:

```markdown
---
language:
- nah
- es
tags:
- translation
- nllb
license: cc-by-nc-4.0
---

# Traductor Náhuatl-Español v4

Modelo de traducción Náhuatl Central → Español.

## Métricas
- BLEU: 22.74
- Corpus: 18,173 pares
- Modelo base: NLLB-200-distilled-1.3B
```

### Crear Hugging Face Space

#### 1. Crear Space

1. Ir a https://huggingface.co/spaces
2. Click "Create new Space"
3. Configuración:
   - Name: `traductor-nahuatl-v4`
   - SDK: Gradio
   - Hardware: CPU basic
   - Visibility: Public

#### 2. Subir Archivos

Subir a la raíz del Space:

**`app.py`:**
```python
import os
import gradio as gr
from transformers import pipeline

MODEL_ID = os.getenv("HUB_MODEL_ID", "tu-usuario/nllb-ncx-es-v4")

translator = pipeline(
    "translation",
    model=MODEL_ID,
    src_lang="nah_Latn",
    tgt_lang="spa_Latn",
    device=-1
)

def translate(text, max_length=128):
    if not text.strip():
        return ""
    try:
        result = translator(text, max_length=int(max_length))
        return result[0]['translation_text']
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# Traductor Náhuatl → Español (v4)")
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Texto en Náhuatl", lines=5)
            max_len = gr.Slider(16, 256, value=128, label="Longitud máxima")
            btn = gr.Button("Traducir", variant="primary")
        with gr.Column():
            out = gr.Textbox(label="Traducción en Español", lines=5)
    btn.click(translate, inputs=[inp, max_len], outputs=out)

if __name__ == "__main__":
    demo.launch()
```

**`requirements.txt`:**
```
transformers==4.45.0
torch==2.0.0
gradio==4.44.0
sentencepiece==0.1.99
```

#### 3. Deployment Automático

El Space se construye automáticamente. Ver progreso en pestaña "Logs".

**Tiempo estimado:** 5-10 minutos

## Uso del Modelo

### Desde Python

```python
from transformers import pipeline

translator = pipeline(
    "translation",
    model="tu-usuario/nllb-ncx-es-v4",
    src_lang="nah_Latn",
    tgt_lang="spa_Latn"
)

result = translator("Cualli tonalli")
print(result[0]['translation_text'])  # "Buenos días"
```

### Desde Hugging Face Space

Acceder a: `https://huggingface.co/spaces/tu-usuario/traductor-nahuatl-v4`

## Composición del Corpus v4

| Fuente | Pares | Porcentaje | Tipo | Método |
|--------|-------|------------|------|--------|
| Axolotl + Tatoeba | 9,156 | 50.3% | Literario | Descarga API |
| JW.org | 5,357 | 29.4% | Religioso | Corpus previo |
| Diccionario Puebla | 3,166 | 17.4% | Lexicográfico | Extracción PDF |
| Diccionario Hueyapan | 388 | 2.1% | Lexicográfico | Extracción PDF |
| INALI | 143 | 0.8% | Educativo | Extracción PDF |
| NHC | 6 | <0.1% | Secular | Extracción PDF |
| **TOTAL** | **18,173** | **100%** | - | - |

## Especificaciones de Scripts

### `extract_dictionary_puebla.py`

**Entrada:** `data/raw/pdfs/diccionario_nahuatl_norte_puebla.md`  
**Salida:** `data/processed/dictionary_puebla.jsonl`  
**Proceso:**
- Parsea formato Markdown del diccionario
- Extrae entradas náhuatl-español
- Valida formato y longitud
- Genera JSONL con metadatos

### `extract_dictionary_hueyapan.py`

**Entrada:** `data/raw/pdfs/nahuatl_hueyapan.md`  
**Salida:** `data/processed/dictionary_hueyapan.jsonl`  
**Proceso:** Similar a extract_dictionary_puebla.py

### `extract_inali_books.py`

**Entrada:** `data/raw/pdfs/libro_nahuatl_2017.md`, `lectura_del_nahuatl.md`  
**Salida:** `data/processed/inali_books.jsonl`  
**Proceso:**
- Extrae textos paralelos de libros educativos
- Alinea párrafos náhuatl-español
- Segmenta en oraciones

### `extract_nhc_parallels.py`

**Entrada:** `data/raw/pdfs/NHC_nahuatl.md`, `NHC_esp.md`  
**Salida:** `data/processed/nhc_parallels.jsonl`  
**Proceso:**
- Alinea párrafos de revistas paralelas
- Valida ratio de longitud
- Expande a nivel de oración

### `download_corpus_FINAL.py`

**Salida:** `data/processed/parallel_ncx_es_EXPANDED.jsonl`  
**Proceso:**
- Descarga corpus Axolotl desde GitHub
- Descarga corpus Tatoeba vía API
- Filtra duplicados con JW.org
- Combina y normaliza
- Guarda en JSONL

### `build_corpus_v3.py`

**Entradas:** Todos los archivos en `data/processed/*.jsonl` y `data/raw/jw_org/parallel_ncx_es.jsonl`  
**Salidas:** `data/processed/corpus_v3.jsonl`, `data/train_test_splits/*.jsonl`  
**Proceso:**
1. Carga todas las fuentes disponibles
2. Normaliza texto (espacios, encoding)
3. Deduplica (case-insensitive)
4. Genera splits 80/10/10 con semilla 42
5. Guarda resultados y estadísticas

### `validate_corpus_v3.py`

**Entrada:** `data/processed/corpus_v3.jsonl`  
**Salida:** Reporte en consola  
**Métricas:**
- Estadísticas básicas (total, longitudes)
- Distribución por fuente/dominio
- Detección de problemas (cortos, largos, ratios)
- Vocabulario único
- Ejemplos por dominio

## Estructura de Datos

### Formato JSONL

Cada línea es un objeto JSON:

```json
{
  "source": "Cualli tonalli",
  "target": "Buenos días",
  "source_lang": "ncx",
  "target_lang": "es",
  "domain": "conversational",
  "metadata": {
    "original_source": "tatoeba",
    "subdomain": "greetings",
    "pair_type": "sentence"
  }
}
```

### Campos Requeridos

- `source`: Texto en Náhuatl (string)
- `target`: Texto en Español (string)
- `source_lang`: "ncx" (string)
- `target_lang`: "es" (string)

### Campos Opcionales

- `domain`: Tipo de dominio (string)
- `metadata`: Metadatos adicionales (object)

## Troubleshooting

### Error: "No module named 'transformers'"

```bash
pip install -r requirements.txt
```

### Error: "FileNotFoundError" en scripts de extracción

Verificar que los archivos `.md` existan en `data/raw/pdfs/`. Si faltan, se requiere conversión previa de PDFs.

### Error: "ConnectionError" al descargar corpus

Verificar conexión a internet. Si no hay internet, usar Opción 2 (corpus parcial).

### Error: "CUDA out of memory" en entrenamiento

Reducir `per_device_train_batch_size` a 2 o usar GPU más grande.

### Error: "Model not found" en Space

Verificar que:
1. El modelo esté público en Hugging Face Hub
2. El `MODEL_ID` en `app.py` sea correcto
3. El modelo tenga todos los archivos necesarios (config.json, model.safetensors, etc.)

### Space no carga

Revisar logs en pestaña "Logs" del Space. Errores comunes:
- Dependencias incorrectas en `requirements.txt`
- Modelo no accesible
- Errores de sintaxis en `app.py`

## Referencias

- **Modelo base:** [NLLB-200-distilled-1.3B](https://huggingface.co/facebook/nllb-200-distilled-1.3B)
- **Paper NLLB:** [No Language Left Behind](https://arxiv.org/abs/2207.04672)
- **Axolotl Corpus:** [GitHub Repository](https://github.com/ivanvladimir/corpus_paralelo_axolotl)
- **Hugging Face Transformers:** [Documentación](https://huggingface.co/docs/transformers)

## Licencia

CC-BY-NC 4.0 (Uso no comercial)

## Versión

4.0 - Octubre 2025
