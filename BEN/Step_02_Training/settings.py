"""Step 02 settings. Step 1 is expected immediately beside this folder."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEP1_DATA = ROOT.parent / 'Step_01_Labeling' / 'data'
RESULTS = ROOT / 'results'
TEST_IMAGES = ROOT / 'data' / 'test_images'

# Default: continue the existing one-class segmentation research.
# To train a detector instead, set TRAIN_TASK='detect', BASE_MODEL='yolo26n.pt'.
TRAIN_TASK = 'segment'
BASE_MODEL = 'yolo26n-seg.pt'
EPOCHS = 80
IMAGE_SIZE = 960
BATCH = 4
# Set to 'cpu' or '0' if necessary; None selects CUDA GPU 0 when available.
DEVICE = None

# Tiling makes small holds in large wall photographs easier to learn.
TILE_SIZE = 1024            # pixels; images smaller than this stay whole
TILE_OVERLAP = 0.20         # fraction; crop overlap is kept within the same split
MIN_VISIBLE_FRACTION = 0.45
MAX_NEGATIVE_TILES_PER_IMAGE = 3
SEED = 42
# Ignore empty labels unless intentionally using true negative photographs.
INCLUDE_EMPTY_LABELS = False

# Set this higher/lower if inference gives too many or too few boxes.
TEST_CONFIDENCE = 0.23
TEST_IMAGE_LIMIT = 100

# Use the same tiling for Step 2 test comparisons as the Step 1 annotator.
TEST_TILE_SIZE = 768
TEST_TILE_OVERLAP = .25
