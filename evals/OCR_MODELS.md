# Stage-1 OCR models, measured

Whether a **two-stage** extraction pipeline beats the single shot the vision
path uses today:

```
two-stage:    PDF → OCR/layout model → text and tables → extraction → JSON
today:        PDF → general VLM → JSON
```

Measured with `python -m evals.ocr_recall data/real --model <model>`. Ground
truth is the deterministic parse of each invoice's own XML. The question is
recall, not accuracy: **if the value is in the text a later stage can extract
it; if it is absent nothing downstream can recover it.**

## Results

Four real supplier invoices, 200 DPI, GTX 1660 Ti.

| field | `granite-docling:258m` | `PaddleOCR-VL-1.6` |
|---|---:|---:|
| `uuid` | 2/4 | **4/4** |
| `rfc_emisor` | 3/4 | 3/4 |
| `rfc_receptor` | 3/4 | **4/4** |
| `subtotal` | 3/4 | **4/4** |
| `total` | **0/4** | **1/4** |
| `line_amounts` | 3/4 | **4/4** |
| median latency | 17.4 s | 20.5 s |

Both drop the number an accounts-payable process exists to check.

## What each result means

**granite-docling (258M, 521 MB, Ollama).** The Spanish OCR is genuinely good
for the size — RFCs, legal names, régimen fiscal, uso CFDI, even the digital
seal — and it is four times faster than `qwen2.5vl:3b`. It misses the totals
block on every invoice. Three different prompts, same 0/4, so this is the
model and not the instruction. Ruled out.

**PaddleOCR-VL-1.6 (~0.9B, 1.8 GB, llama.cpp b8110+).** Clearly better:
`uuid` goes 2/4 to 4/4, and the UUID is the hardest field on the page and the
one that defines fiscal identity. Still 1/4 on `total`.

Asking explicitly for the totals box moved `total` to 2/4 and pulled three
other fields *down*. At four invoices that is noise.

**The likely cause, untested.** PaddleOCR-VL's published 96.33% on
OmniDocBench comes from its own pipeline: layout detection first, then
recognition per cropped region. Driving it single-shot over a whole page
through `llama-server` skips that first stage, so region selection is left to
the model. The totals box on a CFDI is a small right-aligned block, exactly
the kind of region a whole-page pass drops. Running the real PaddleOCR
pipeline instead of one-shot llama.cpp is the next experiment, and it may be
the whole difference.

**granite-4.0-3b-vision** is built for precisely this task — key-value pair
extraction, tables to JSON, 85.5% exact-match on VAREX — but IBM's model card
states it is trained on English instructions only and degrades on other
languages. A CFDI is in Spanish. Not tested for that reason.

## The limit on all of the above

**Four invoices.** Every number in that table moves by 25% for one document.
The comparison is directional at best, and no threshold or model choice
should be made on it. 30 to 50 PDF/XML pairs is where these become
measurements.

## Reproducing

Ollama models:

```bash
docker compose --profile embeddings up -d
docker compose exec embeddings ollama pull ibm/granite-docling:258m
python -m evals.ocr_recall data/real --model ibm/granite-docling:258m
```

PaddleOCR-VL needs llama.cpp, because Ollama does not carry it — the separate
mmproj architecture is why. Prebuilt Vulkan binaries avoid a CUDA build and
work on NVIDIA; an arm64 build exists for the Orin:

```bash
# llama.cpp b8110 or later
curl -sL -o llama.tar.gz https://github.com/ggml-org/llama.cpp/releases/download/b10173/llama-b10173-bin-ubuntu-vulkan-x64.tar.gz
tar xzf llama.tar.gz

# ~1.8 GB of weights
curl -sL -o PaddleOCR-VL-1.6-GGUF.gguf https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF/resolve/main/PaddleOCR-VL-1.6-GGUF.gguf
curl -sL -o PaddleOCR-VL-1.6-GGUF-mmproj.gguf https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF/resolve/main/PaddleOCR-VL-1.6-GGUF-mmproj.gguf

LD_LIBRARY_PATH=. ./llama-server \
  -m PaddleOCR-VL-1.6-GGUF.gguf \
  --mmproj PaddleOCR-VL-1.6-GGUF-mmproj.gguf \
  --temp 0 --port 8080 -ngl 99 -c 16384

python -m evals.ocr_recall data/real \
  --model paddleocr-vl --base-url http://localhost:8080/v1 --prompt "OCR:"
```

`llama-server` speaks the same OpenAI-compatible API as Ollama, so the
harness needs no change to point at it. That is what the provider seam is
for.

Transcriptions are cached under `data/.ocr_cache/`, so re-running a model and
prompt already measured costs nothing. `data/` is gitignored — the cache
holds full invoice text and must never be committed.
