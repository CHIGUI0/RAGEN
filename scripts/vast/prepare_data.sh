#!/bin/bash
# Prepare search benchmark data on Vast.ai machine:
#   1. Download wiki corpus + FAISS index shards (~74 GB)
#   2. Merge index shards → e5_Flat.index
#   3. Convert wiki-18.jsonl → corpus.json
#   4. Prepare HotpotQA parquet data

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_step() { echo -e "${BLUE}[Step] ${1}${NC}"; }
print_ok()   { echo -e "${GREEN}[OK]   ${1}${NC}"; }

if [ ! -f "train.py" ]; then
    echo "Please run this script from the RAGEN repo root."
    exit 1
fi

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate ragen

DATA_DIR="./search_data"
INDICES_DIR="${DATA_DIR}/prebuilt_indices"
WIKI_DIR="${DATA_DIR}/wikipedia"

# ========== 1. Download corpus + index shards ==========
print_step "Downloading search index data (corpus + FAISS shards)..."
python scripts/download_search_index.py --data_dir "$DATA_DIR"

# ========== 2. Merge index shards ==========
INDEX_FILE="${INDICES_DIR}/e5_Flat.index"
if [ -f "$INDEX_FILE" ]; then
    print_ok "e5_Flat.index already exists ($(du -h "$INDEX_FILE" | cut -f1))"
else
    print_step "Merging index shards → e5_Flat.index..."
    if [ -f "${INDICES_DIR}/part_aa" ] && [ -f "${INDICES_DIR}/part_ab" ]; then
        cat "${INDICES_DIR}/part_aa" "${INDICES_DIR}/part_ab" > "$INDEX_FILE"
        print_ok "Created e5_Flat.index ($(du -h "$INDEX_FILE" | cut -f1))"
        # Remove shards to save disk
        rm -f "${INDICES_DIR}/part_aa" "${INDICES_DIR}/part_ab"
        print_ok "Removed index shards to save space"
    else
        echo "ERROR: Index shards not found in ${INDICES_DIR}"
        exit 1
    fi
fi

# ========== 3. Convert wiki-18.jsonl → corpus.json ==========
CORPUS_FILE="${INDICES_DIR}/corpus.json"
WIKI_JSONL="${WIKI_DIR}/wiki-18.jsonl"

if [ -f "$CORPUS_FILE" ]; then
    print_ok "corpus.json already exists ($(du -h "$CORPUS_FILE" | cut -f1))"
else
    print_step "Converting wiki-18.jsonl → corpus.json..."
    if [ ! -f "$WIKI_JSONL" ]; then
        echo "ERROR: ${WIKI_JSONL} not found"
        exit 1
    fi
    python3 -c "
import json, sys
from tqdm import tqdm

input_path = '${WIKI_JSONL}'
output_path = '${CORPUS_FILE}'

print(f'Reading {input_path}...')
corpus = []
with open(input_path, 'r') as f:
    for line in tqdm(f, desc='Loading wiki-18.jsonl'):
        line = line.strip()
        if not line:
            continue
        doc = json.loads(line)
        # Extract text content (handle different possible field names)
        text = doc.get('text', doc.get('contents', doc.get('content', '')))
        title = doc.get('title', '')
        if title and text:
            corpus.append(f'{title} {text}')
        elif text:
            corpus.append(text)

print(f'Writing {len(corpus)} documents to {output_path}...')
with open(output_path, 'w') as f:
    json.dump(corpus, f)
print(f'Done! corpus.json = {len(corpus)} docs')
"
    print_ok "Created corpus.json ($(du -h "$CORPUS_FILE" | cut -f1))"
fi

# ========== 4. Prepare HotpotQA parquet ==========
print_step "Preparing HotpotQA parquet data..."
python scripts/prepare_search_data.py --output_dir data/search

# ========== Verify ==========
echo ""
echo "=== Data verification ==="
for f in "$INDEX_FILE" "$CORPUS_FILE" "data/search/train.parquet" "data/search/val.parquet"; do
    if [ -f "$f" ]; then
        print_ok "$f  ($(du -h "$f" | cut -f1))"
    else
        echo "MISSING: $f"
    fi
done

echo ""
print_ok "Data preparation complete!"
echo "Next step:  bash scripts/vast/run_all_search.sh"
