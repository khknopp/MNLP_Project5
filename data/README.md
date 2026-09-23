# TED data preparation

Run commands from the repository root. These examples use PowerShell on Windows
and assume Python is available. Downloading and extracting the corpus does not
train a translation model.

## 1. Download the archive

The project uses `data/TED.tar` from the Hugging Face dataset
[`davidstap/ted_talks`](https://huggingface.co/datasets/davidstap/ted_talks).
Hugging Face hosts the download; subsequent preparation uses your local copy.

```powershell
New-Item -ItemType Directory -Force data
curl.exe -L --fail --retry 3 -C - -o data/TED.tar "https://huggingface.co/datasets/davidstap/ted_talks/resolve/main/data/TED.tar"
```

If the archive is already downloaded, skip this step. On Linux/macOS, use
`mkdir -p data` and `curl` instead of the PowerShell directory command and `curl.exe`.

## 2. Extract the original text

```powershell
python src/preprocessing/align.py extract
```

This extracts English (`en`), Azerbaijani (`az`), Belarusian (`be`), Turkish (`tr`),
Ukrainian (`uk`) and Russian (`ru`) for the `train`, `dev` and `test` splits:
18 text files in total. Existing files are skipped; use `extract --force` only
when you intend to replace them from the archive.

**Blank lines are alignment placeholders.** A missing translation must not shift
all subsequent sentences. Keep these blank lines in `raw/`; extraction is not
text cleaning.

## 3. Prepare datasets

Before generating parallel files, save the committed reference manifest for
independent verification. This reads Git's reference, not a locally regenerated
manifest:

```powershell
python -c "from pathlib import Path; import subprocess; Path('data/reference_manifest.json').write_bytes(subprocess.check_output(['git', 'show', 'HEAD:data/processed/parallel/manifest.json']))"
```

Then generate the prepared datasets:

```powershell
# Translation pairs and original position numbers
python src/preprocessing/align.py emit

# Triples for translation through a helper language
python src/preprocessing/align.py pivot

# Separate language texts for vocabulary training
python src/preprocessing/prep_data.py
```

| Output | Use |
| :--- | :--- |
| `processed/parallel/` | Aligned pairs for translation training and evaluation |
| `processed/pivot/` | Aligned triples for pivot experiments |
| `processed/vocab_training/` | Text for training custom tokenizers only |

`prep_data.py` reads the archive directly and removes blank lines separately for
each language. Do not match its output files by line number to make translation
pairs. Use `align.py` or `src.data.get_language_pairs` for aligned data.

## 4. Verify the data

Check the generated files against the saved team reference:

```powershell
python src/preprocessing/verify.py --manifest data/reference_manifest.json --dir data/processed/parallel
```

The default command below checks against the local manifest. It is useful for
spotting changes after generation, but cannot independently establish agreement
with the team because `align.py emit` regenerates that manifest:

```powershell
python src/preprocessing/verify.py
```

### Windows line endings

The current writer can produce CRLF line endings on Windows, while the committed
reference hashes may describe LF files. Byte hashes then differ even when sentence
text and alignment are identical. In this checkout, all 81 parallel files matched
the committed hashes after CRLF-to-LF normalization during verification.

If reference verification fails, inspect the differences; do not assume every
mismatch is a line-ending issue or replace the team reference just to pass the
check. An updated manifest is not a dataset change to commit automatically.

## 5. Choose the next step

**NLLB experiments:** use NLLB's pretrained tokenizer. The baseline notebook
prepares its JSONL inputs under `data/processed/nllb/`. Custom vocabularies are not
required for this route.

**Custom vocabulary experiments:** install the project dependencies with `uv sync`,
then train the vocabulary variants:

```powershell
uv run python src/preprocessing/train_vocabs.py --model unigram
uv run python src/preprocessing/train_vocabs.py --model bpe
uv run python src/preprocessing/tokenize_corpus.py
```

Tokenizers are written under `artifacts/vocabularies/`; tokenized text goes to
`data/processed/tokenized/`. Vocabulary training also updates `index.json`, so
preserve the team's reference index before retraining if you want an independent
hash comparison.

## Resulting layout

```text
data/
|-- README.md
|-- TED.tar
|-- reference_manifest.json     # Local copy of the committed reference
|-- raw/
|   |-- train/                 # en.txt, az.txt, be.txt, tr.txt, uk.txt, ru.txt
|   |-- dev/
|   `-- test/
`-- processed/
    |-- parallel/
    |-- pivot/
    |-- vocab_training/
    |-- tokenized/             # Created by optional tokenization
    `-- nllb/                  # Created by the NLLB notebook
```

Generated corpus files stay local and are ignored by Git. This guide and the
parallel reference manifest are tracked. See the [main README](../README.md)
for the full repository layout.
