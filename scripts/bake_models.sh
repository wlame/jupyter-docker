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
# Usage: bake_models.sh <target>     # vision | nlp | speech | face | full
# =============================================================================
set -euo pipefail

YOLO_URL="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
YOLO_SHA256="f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"

# NLTK resources example 14 requires (matches its download list).
NLTK_RESOURCES=(
    punkt punkt_tab stopwords vader_lexicon wordnet
    averaged_perceptron_tagger averaged_perceptron_tagger_eng
)

# Run python from the synced project venv.
py() { uv run --no-project python "$@"; }

bake_vision() {
    local dest="${HOME}/.cache/ultralytics/yolov8n.pt"
    echo "→ YOLOv8n weights"
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

bake_speech() {
    echo "→ Whisper tiny"
    py -c "import whisper; whisper.load_model('tiny')"
}

bake_face() {
    echo "→ face-alignment detector + landmark nets (s3fd, 2DFAN)"
    py -c "import face_alignment; face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu')"
}

target="${1:?usage: bake_models.sh <vision|nlp|speech|face|full>}"
case "${target}" in
    vision) bake_vision ;;
    nlp)    bake_nlp ;;
    speech) bake_speech ;;
    face)   bake_face ;;
    full)   bake_vision; bake_nlp; bake_speech; bake_face ;;
    *)      echo "no models to bake for target: ${target}"; exit 0 ;;
esac
echo "✓ model baking complete for ${target}"
