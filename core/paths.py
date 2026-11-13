"""Every fixed path in the project, in one place. Import these instead of building paths from __file__.

    from core.paths import ROOT, CONFIG, CALIBRATION, URDF
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Configuration (YAML, read through core/config.py)
CONFIG = ROOT / "config"
ARMS_FILE = CONFIG / "arms.yaml"
PERCEPTION_FILE = CONFIG / "perception.yaml"
GAMES_CONFIG = CONFIG / "games"
CALIBRATION = CONFIG / "calibration"                  # every calibration value of this setup
CAMERA_CALIBRATION_FILE = CALIBRATION / "camera.yaml"
CAMERA_TRANSFORM_FILE = CALIBRATION / "camera_transform.npy"   # derived from camera.yaml, never edited by hand
COLOURS_CALIBRATION_FILE = CALIBRATION / "colours.yaml"


def arm_calibration_dir(arm: str) -> Path:
    """config/calibration/<arm>/: that arm's motor calibration (motors.yaml) and fine joint offsets."""
    return CALIBRATION / arm


# Models: object meshes, the robot description, trained YOLO weights
MODELS = ROOT / "models"
OBJECTS_DIR = MODELS / "objects"
URDF = MODELS / "robot" / "so101" / "robot.urdf"
YOLO_DIR = MODELS / "yolo"
YOLO_RUNS = ROOT / "runs" / "segment"                 # where `ultralytics` training writes its runs

# Run outputs (pictures, logs); not tracked
OUTPUTS = ROOT / "outputs"
