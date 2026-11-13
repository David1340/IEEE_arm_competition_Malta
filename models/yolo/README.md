# models/yolo/

Trained YOLO segmentation weights (`*.pt`). They are large, so git ignores them: copy `best.pt` here on each
computer (or leave it in `runs/segment/<run>/weights/`, where training writes it).

Which weights the games use is set in `config/perception.yaml` (`yolo.weights`): a path, or `latest` = the newest
`runs/segment/*/weights/best.pt`, else the newest `models/yolo/*.pt`. `--weights` on the command line overrides it.
