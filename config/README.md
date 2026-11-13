# config/

Every setting and every calibration value of the setup, in YAML. Code reads them through `core/` (never by path
from a script): `core/arms.py` (arms.yaml), `games/rules.py` (games/), `perception/detection.py`
(perception.yaml), `hardware/joint_offsets.py` and `calibration/` (calibration/). Each file explains itself in its
header comment; tools that write a file keep that header.

| file | contents | written by |
|---|---|---|
| `arms.yaml` | the four arms: ports, USB IDs, leader pairing, base positions, home pose, handoff points, LeRobot types | you; `01_setup_usb.py` (usb, port); `02_calibrate.py camera touch` / `offsets --arm` (base) |
| `perception.yaml` | camera ROI, YOLO weights and classes, colour hues | you; `roi.py select` |
| `grasping.yaml` | per object type: grasp postures to try, least finger cover for top-down grasps, squeeze | you |
| `teleop.yaml` | network teleop addresses | you |
| `games/<game>.yaml` | each game's rules (rounds, which bin each object goes in) | you |
| `calibration/camera.yaml` | camera -> TABLE frame | `02_calibrate.py camera ...` |
| `calibration/<arm>/motors.yaml` | the arm's LeRobot motor calibration (versioned copy) | `02_calibrate.py motors` / `sync pull` |
| `calibration/<arm>/joint_offsets.yaml` | the follower's fine joint offsets | `02_calibrate.py offsets` |
| `calibration/colours.yaml` | fitted colour hues (optional) | `python -m perception.colour --calibrate` |

`calibration/camera_transform.npy` is derived from `camera.yaml` on every save; git ignores it.

YAML notes: keep USB IDs quoted (`"0483"` unquoted would be read as a number); `python 02_calibrate.py status`
checks that everything loads.
