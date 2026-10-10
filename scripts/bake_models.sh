#!/usr/bin/env bash
# =============================================================================
# Pre-bake model weights into the image so example smoke tests run offline.
#
# Runs at BUILD time (network available) as the jupyter user. Each model lands
# in its library's DEFAULT cache location under /home/jupyter, so nothing needs
# a runtime environment variable — the libraries find their caches on their own.
#
# Only the standalone weight file we can pin (YOLO) is checksum-verified here;
# the library-managed caches (NLTK, HuggingFace, Whisper, torch hub) rely on
# each library's own integrity checks.
#
# Usage: bake_models.sh <target>     # vision | nlp | genai | speech | face | full
# =============================================================================
set -euo pipefail

YOLO_URL="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"
YOLO_SHA256="9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"

# NLTK resources example 14 requires (matches its download list).
NLTK_RESOURCES=(
    punkt punkt_tab stopwords vader_lexicon wordnet
    averaged_perceptron_tagger averaged_perceptron_tagger_eng
)

# Run python from the synced project venv.
py() { uv run --no-project python "$@"; }

bake_vision() {
    local dest="${HOME}/.cache/ultralytics/yolo26n.pt"
    echo "→ YOLO26n weights"
    mkdir -p "$(dirname "${dest}")"
    curl -fsSL --retry 3 -o "${dest}" "${YOLO_URL}"
    echo "${YOLO_SHA256}  ${dest}" | sha256sum -c -
}

bake_nlp() {
    echo "→ NLTK data"
    py -m nltk.downloader -d "${HOME}/nltk_data" "${NLTK_RESOURCES[@]}"
    echo "→ sentence-transformers all-MiniLM-L6-v2"
    py -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
}

# A small instruction-tuned language model for the RAG example (Apache-2.0,
# about 270 MB). Pinned to a revision; example 37 loads the same one.
SMOLLM_REPO="HuggingFaceTB/SmolLM2-135M-Instruct"
SMOLLM_REVISION="12fd25f77366fa6b3b4b768ec3050bf629380bac"

bake_genai() {
    echo "→ ${SMOLLM_REPO} (weights, config, tokenizer)"
    py -c "from huggingface_hub import snapshot_download; snapshot_download('${SMOLLM_REPO}', revision='${SMOLLM_REVISION}', allow_patterns=['*.json', 'model.safetensors', 'merges.txt'], ignore_patterns=['onnx/*'])"
}

bake_speech() {
    echo "→ Whisper tiny"
    py -c "import whisper; whisper.load_model('tiny')"
}

bake_face() {
    echo "→ face-alignment detector + landmark nets (s3fd, 2DFAN)"
    py -c "import face_alignment; face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu')"
}

target="${1:?usage: bake_models.sh <vision|nlp|genai|speech|face|full>}"
case "${target}" in
    vision) bake_vision ;;
    nlp)    bake_nlp ;;
    genai)  bake_genai ;;
    speech) bake_speech ;;
    face)   bake_face ;;
    full)   bake_vision; bake_nlp; bake_genai; bake_speech; bake_face ;;
    *)      echo "no models to bake for target: ${target}"; exit 0 ;;
esac
echo "✓ model baking complete for ${target}"
