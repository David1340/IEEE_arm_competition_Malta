"""Step 02 — Train YOLO hold segmentation and compare YOLO masks vs YOLO-box/SAM masks.

From this folder: python main.py             (four-option menu)
Or:               python main.py prepare | train | test | update

No dataset edits are made in Step 01; only the optional explicit 'update' action
changes Step_01_Labeling/data/best.pt (after backing it up).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
import settings as cfg

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def paths():
    base = Path(cfg.STEP1_DATA)
    return base / 'images', base / 'labels', base / 'best.pt'


def check_source():
    photos, labels, _ = paths()
    if not photos.is_dir():
        raise RuntimeError(f'Step 1 photographs folder not found: {photos}\n'
                           'Place Step_01_Labeling and Step_02_Training beside each other, '
                           'or edit STEP1_DATA in settings.py.')
    if not labels.is_dir():
        raise RuntimeError(f'Step 1 labels folder not found: {labels}')
    return photos, labels


def area(pts):
    return abs(sum(x*y2-x2*y for (x,y),(x2,y2) in zip(pts,pts[1:]+pts[:1]))) * .5


def read_polygons(path):
    """Read and strictly validate Step 1's class-0 normalized YOLO polygons."""
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    polygons=[]
    for number, line in enumerate(lines, 1):
        parts=line.split()
        if not parts: continue
        if parts[0] != '0' or len(parts)<7 or len(parts)%2 != 1:
            raise ValueError(f'{path.name}, line {number}: expected class 0 and >= 3 x,y pairs')
        try:
            numbers=[float(x) for x in parts[1:]]
        except ValueError as exc:
            raise ValueError(f'{path.name}, line {number}: non-numeric coordinates') from exc
        if not all(math.isfinite(v) and -.00001<=v<=1.00001 for v in numbers):
            raise ValueError(f'{path.name}, line {number}: coordinates must be in 0..1')
        poly=[(min(1.,max(0.,numbers[i])), min(1.,max(0.,numbers[i+1])))
              for i in range(0,len(numbers),2)]
        if area(poly)<1e-9:
            raise ValueError(f'{path.name}, line {number}: collapsed/zero-area polygon')
        polygons.append(poly)
    return polygons


def photo_records():
    photos,labels=check_source()
    files=sorted([p for p in photos.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS],
                 key=lambda p:p.name.casefold())
    stems=defaultdict(list)
    for photo in files:stems[photo.stem.lower()].append(photo)
    ambiguous=[names for names in stems.values() if len(names)>1]
    if ambiguous:
        raise RuntimeError('Step 1 has photographs with the same stem and different extensions: '
                           + ', '.join('/'.join(n.name for n in group) for group in ambiguous)
                           + '\nStep 1 uses the same label name for both; rename one before training.')
    records=[];unlabelled=[];empty_labels=[]
    for photo in files:
        label=labels/(photo.stem+'.txt')
        if not label.is_file():
            unlabelled.append(photo.name)
            continue  # An unlabeled image is NOT automatically a negative example.
        with Image.open(photo) as im:
            if int(im.getexif().get(274,1) or 1)!=1:
                raise RuntimeError(f'{photo.name} has uncorrected EXIF orientation. '
                                   'Open it and save it in Step 1 so labels and pixels align.')
            w,h=im.size
        if w<64 or h<64:raise ValueError(f'{photo.name}: image too small')
        polys=read_polygons(label)
        if not polys and not cfg.INCLUDE_EMPTY_LABELS:
            empty_labels.append(photo.name)
            continue
        digest=hashlib.sha256(photo.read_bytes()).hexdigest()
        records.append(dict(image=photo,label=label,polygons=polys,w=w,h=h,digest=digest))
    return records,unlabelled,empty_labels


def split_photos(records):
    """Split ORIGINAL photographs before tiling, keeping exact-duplicate files together."""
    rng=random.Random(cfg.SEED)
    by_digest=defaultdict(list)
    for record in records:by_digest[record['digest']].append(record)
    groups=list(by_digest.values())
    positive=[g for g in groups if any(r['polygons'] for r in g)]
    negative=[g for g in groups if not any(r['polygons'] for r in g)]
    rng.shuffle(positive);rng.shuffle(negative)
    if len(groups)<2 or len(positive)<2:
        raise ValueError('Need at least TWO independently photographed, labelled images containing holds. '
                         'One image is useful for editing/debugging but cannot give a meaningful train/val split.')
    use_test=(len(groups)>=8 and len(positive)>=3)
    val_count=max(1,round(len(groups)*.2))
    test_count=max(1,round(len(groups)*.15)) if use_test else 0
    # Reserve a *positive* group for validation and (when possible) test.
    selected={'train':[], 'val':[positive.pop()], 'test':[]}
    if use_test:selected['test'].append(positive.pop())
    pool=positive+negative
    rng.shuffle(pool)
    for target,count in [('val',val_count),('test',test_count)]:
        while len(selected[target])<count and pool:
            # Never move the last remaining positive training group.
            idx=next((i for i,g in enumerate(pool)
                      if (not any(r['polygons'] for r in g) or
                          sum(any(r['polygons'] for r in x) for x in pool)>1)),None)
            if idx is None:break
            selected[target].append(pool.pop(idx))
    selected['train']=pool
    if not any(any(r['polygons'] for r in group) for group in selected['train']):
        raise ValueError('No positive training photographs after splitting.')
    return {name:[r for group in gs for r in group] for name,gs in selected.items()}


def start_positions(length,tile,overlap):
    if length<=tile:return [0]
    stride=max(1,round(tile*(1-overlap)))
    values=list(range(0,length-tile+1,stride))
    if values[-1]!=length-tile:values.append(length-tile)
    return values


def clipped_polygon(points, x0,y0,x1,y1):
    """Sutherland–Hodgman clip against a rectangular image tile."""
    pts=list(points)
    for axis,limit,keep_high in [(0,x0,True),(0,x1,False),(1,y0,True),(1,y1,False)]:
        if not pts:break
        out=[];previous=pts[-1]
        def inside(point):return point[axis]>=limit if keep_high else point[axis]<=limit
        def intersect(a,b):
            delta=b[axis]-a[axis]
            factor=(limit-a[axis])/delta if abs(delta)>1e-12 else 0.
            return (a[0]+factor*(b[0]-a[0]),a[1]+factor*(b[1]-a[1]))
        for current in pts:
            if inside(current):
                if not inside(previous):out.append(intersect(previous,current))
                out.append(current)
            elif inside(previous):out.append(intersect(previous,current))
            previous=current
        pts=out
    return pts


def tile_boxes(polys,w,h,x,y,tw,th):
    boxes=[]
    for poly in polys:
        pixel=[(xx*w,yy*h) for xx,yy in poly]
        original_area=area(pixel)
        if original_area<10:continue
        cropped=clipped_polygon(pixel,x,y,x+tw,y+th)
        if len(cropped)<3 or area(cropped)/original_area<cfg.MIN_VISIBLE_FRACTION:
            continue
        xa=min(t[0] for t in cropped);xb=max(t[0] for t in cropped)
        ya=min(t[1] for t in cropped);yb=max(t[1] for t in cropped)
        if xb-xa<5 or yb-ya<5:continue
        bx=(xa+xb)/2-x;by=(ya+yb)/2-y
        bw=xb-xa;bh=yb-ya
        boxes.append((bx/tw,by/th,bw/tw,bh/th))
    return boxes



def tile_segments(polys,w,h,x,y,tw,th):
    """Clip Step 1 polygons to a tile and save precise YOLO segmentation labels."""
    segments=[]
    for poly in polys:
        pixel=[(xx*w,yy*h) for xx,yy in poly]
        original_area=area(pixel)
        if original_area<10:continue
        cropped=clipped_polygon(pixel,x,y,x+tw,y+th)
        if len(cropped)<3 or area(cropped)/original_area<cfg.MIN_VISIBLE_FRACTION:continue
        xs=[q[0] for q in cropped];ys=[q[1] for q in cropped]
        if max(xs)-min(xs)<5 or max(ys)-min(ys)<5:continue
        output=[]
        for px,py in cropped:
            output.extend((max(0.,min(1.,(px-x)/tw)),max(0.,min(1.,(py-y)/th))))
        if len(output)>=6:segments.append(tuple(output))
    return segments


def _jpg(image, dest):
    if image.mode!='RGB':image=image.convert('RGB')
    image.save(dest,format='JPEG',quality=94,subsampling=0)


def generate_tiles(records, split, destination, rng):
    out_images=destination/'images'/split
    out_labels=destination/'labels'/split
    out_images.mkdir(parents=True,exist_ok=True);out_labels.mkdir(parents=True,exist_ok=True)
    tiles=0; positives=0; negatives=0; boxes_total=0
    for rec in records:
        with Image.open(rec['image']) as original:
            source=original.convert('RGB')
            w,h=source.size
            choices=[]
            for yy in start_positions(h,cfg.TILE_SIZE,cfg.TILE_OVERLAP):
                for xx in start_positions(w,cfg.TILE_SIZE,cfg.TILE_OVERLAP):
                    tw=min(cfg.TILE_SIZE,w-xx);th=min(cfg.TILE_SIZE,h-yy)
                    boxes=(tile_segments(rec['polygons'],w,h,xx,yy,tw,th) if cfg.TRAIN_TASK=='segment'
                           else tile_boxes(rec['polygons'],w,h,xx,yy,tw,th))
                    choices.append((xx,yy,tw,th,boxes))
            good=[x for x in choices if x[4]]
            empty=[x for x in choices if not x[4]]
            rng.shuffle(empty)
            # Don't swamp scarce positive tiles with blank wall sections.
            if rec['polygons']:
                num_empty=min(cfg.MAX_NEGATIVE_TILES_PER_IMAGE,max(1,round(len(good)*.4)))
            else:
                num_empty=cfg.MAX_NEGATIVE_TILES_PER_IMAGE
            selected=good+empty[:num_empty]
            if rec['polygons'] and not good:
                raise ValueError(f'{rec["image"].name}: all annotated holds vanished from tiles. '
                                 'Increase TILE_SIZE or lower MIN_VISIBLE_FRACTION.')
            for xx,yy,tw,th,boxes in selected:
                # Digest avoids collisions between identically named photos in other folders.
                stem=f'{rec["image"].stem}_{rec["digest"][:10]}_{xx}_{yy}'
                _jpg(source.crop((xx,yy,xx+tw,yy+th)),out_images/f'{stem}.jpg')
                content='\n'.join('0 '+' '.join(f'{v:.6f}' for v in b) for b in boxes)
                (out_labels/f'{stem}.txt').write_text(content+('\n' if content else ''),encoding='utf-8')
                tiles+=1;boxes_total+=len(boxes)
                positives+=bool(boxes);negatives+=not bool(boxes)
    return dict(photos=len(records),tiles=tiles,positive_tiles=positives,
                negative_tiles=negatives,hold_boxes=boxes_total)


def make_preview(dataset, count=12):
    paths_=sorted((dataset/'images'/'train').glob('*.jpg'))
    rng=random.Random(cfg.SEED)
    rng.shuffle(paths_)
    paths_=paths_[:count]
    if not paths_:return
    card_w,card_h=284,220
    cols=3;rows=math.ceil(len(paths_)/cols)
    board=Image.new('RGB',(cols*card_w,rows*card_h),(239,241,245))
    draw=ImageDraw.Draw(board)
    for idx,path in enumerate(paths_):
        with Image.open(path) as img:
            img=img.convert('RGB');w,h=img.size
            label=dataset/'labels'/'train'/(path.stem+'.txt')
            image_draw=ImageDraw.Draw(img)
            for line in label.read_text().splitlines():
                parts=line.split()
                vals=list(map(float,parts[1:]))
                if len(vals)==4:
                    cx,cy,bw,bh=vals
                    image_draw.rectangle(((cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h),
                                         outline=(245,160,40),width=max(2,round(min(w,h)/250)))
                elif len(vals)>=6 and len(vals)%2==0:
                    image_draw.polygon([(vals[i]*w, vals[i+1]*h) for i in range(0,len(vals),2)],
                                       outline=(245,160,40),width=max(2,round(min(w,h)/250)))
            img.thumbnail((card_w-12,card_h-35))
            xx=(idx%cols)*card_w;yy=(idx//cols)*card_h
            board.paste(img,(xx+(card_w-img.width)//2,yy))
            draw.text((xx+6,yy+card_h-25),path.stem[-35:],fill=(20,35,48))
    board.save(dataset/'preview.jpg')


def prepare():
    records,unlabelled,empty_labels=photo_records()
    print(f'Found {len(records)} usable labelled photographs, {len(unlabelled)} without labels, '
          f'and {len(empty_labels)} with empty labels.')
    if empty_labels:
        print('Empty label files skipped to avoid treating unfinished photos as negatives. '
              'Set INCLUDE_EMPTY_LABELS=True in settings.py to include deliberate negative examples.')
    if unlabelled:
        print('Unlabelled images are excluded, NOT treated as blank backgrounds: '
              + ', '.join(unlabelled[:6]) + (' ...' if len(unlabelled)>6 else ''))
    if not records:raise RuntimeError('No labelled photographs. Draw and save polygons in Step 1 first.')
    splits=split_photos(records)
    root=Path(cfg.RESULTS);root.mkdir(parents=True,exist_ok=True)
    dataset=root/'dataset';stage=root/'_dataset_building'
    if stage.exists():shutil.rmtree(stage)
    stage.mkdir(parents=True)
    rng=random.Random(cfg.SEED)
    try:
        info={}
        for key in ('train','val','test'):
            if not splits[key]:continue
            info[key]=generate_tiles(splits[key],key,stage,rng)
            info[key]['files']=[r['image'].name for r in splits[key]]
        if not info['train']['hold_boxes']:
            raise ValueError('No training holds were generated.')
        if not info['val']['hold_boxes']:
            print('WARNING: validation has no positive holds; mAP is not meaningful.')
        # Print a YAML that Ultralytics accepts even on Windows paths.
        data_yaml='path: '+json.dumps(str(dataset.resolve().as_posix()))+'\n'
        data_yaml+='train: images/train\nval: images/val\n'
        if info.get('test'):data_yaml+='test: images/test\n'
        data_yaml+='names:\n  0: speed_handhold\n'
        (stage/'data.yaml').write_text(data_yaml,encoding='utf-8')
        (stage/'summary.json').write_text(json.dumps({
            'source':str(Path(cfg.STEP1_DATA).resolve()),'split_by':'original photo (exact duplicates kept together)',
            'seed':cfg.SEED,'task':cfg.TRAIN_TASK,'settings':{'tile_size':cfg.TILE_SIZE,'tile_overlap':cfg.TILE_OVERLAP},
            'splits':info},indent=2),encoding='utf-8')
        make_preview(stage)
        # Build in a staging folder; preserve last known-good dataset if creation fails.
        if dataset.exists():shutil.rmtree(dataset)
        stage.rename(dataset)
    finally:
        if stage.exists():shutil.rmtree(stage)
    for name,details in info.items():
        print(f'{name.upper():5s}: {details["photos"]} photos, {details["tiles"]} tiles, '
              f'{details["hold_boxes"]} hold instances')
    print(f'\nDataset: {dataset / "data.yaml"}\nPreview: {dataset / "preview.jpg"}')
    if not info.get('test'):
        print('NOTE: No independent test set reserved (too few independent photos). '
              'Use new unlabelled photos for the Test option.')
    print('IMPORTANT: photographs from the same recording session can still leak '
          'if they are different images. Keep related photos together when evaluating.')
    return dataset


def latest_model():
    folder=Path(cfg.RESULTS)/'training'
    found=list(folder.glob('**/weights/best.pt'))
    return max(found,key=lambda p:p.stat().st_mtime) if found else None


def device():
    if cfg.DEVICE is not None:return cfg.DEVICE
    try:
        import torch
        return '0' if torch.cuda.is_available() else 'cpu'
    except ImportError:return 'cpu'


def train(use_previous=False, from_step1=False):
    dataset=Path(cfg.RESULTS)/'dataset'/'data.yaml'
    if not dataset.is_file():
        raise RuntimeError('Dataset not prepared yet. Choose option 1 (Prepare) first.')
    summary=json.loads((dataset.parent/'summary.json').read_text(encoding='utf8'))
    if summary.get('task')!=cfg.TRAIN_TASK:
        raise RuntimeError('Prepared dataset has the wrong task/label format. '
                           'Choose option 1 again to prepare '+cfg.TRAIN_TASK+' labels.')
    modelpath=str(cfg.BASE_MODEL)
    if use_previous and from_step1:
        raise ValueError('Choose either a previous Step 2 run OR the original Step 1 weights.')
    if use_previous:
        previous=latest_model()
        if previous is None:raise RuntimeError('No previous trained best.pt found. Train from pretrained first.')
        modelpath=str(previous)
    elif from_step1:
        step1_model=paths()[2]
        if not step1_model.is_file():raise FileNotFoundError(f'Step 1 weights not found: {step1_model}')
        modelpath=str(step1_model)
    try:from ultralytics import YOLO
    except ImportError as e:raise RuntimeError('Install Ultralytics: python -m pip install ultralytics') from e
    print(f'Model: {modelpath}\nDevice: {device()}\nEpochs: {cfg.EPOCHS}, image size {cfg.IMAGE_SIZE}, batch {cfg.BATCH}')
    model=YOLO(modelpath)
    if model.task!=cfg.TRAIN_TASK:
        raise RuntimeError(f'Expected YOLO {cfg.TRAIN_TASK} weights, got task={model.task!r}. '
                           'Select an appropriate pretrained model or do not continue from incompatible weights.')
    model.train(data=str(dataset),epochs=cfg.EPOCHS,imgsz=cfg.IMAGE_SIZE,batch=cfg.BATCH,
                device=device(),workers=0,seed=cfg.SEED,project=str(Path(cfg.RESULTS)/'training'),
                name='holds',exist_ok=False,patience=20,plots=True)
    best=latest_model()
    if best is None:raise RuntimeError('Training completed, but best.pt was not found in results/training.')
    print(f'\nTrained model: {best}')
    print('Step 1 best.pt has NOT been changed. Choose option 4 to update it.')
    return best


def comparison_paths(source=None):
    """A dedicated folder of NEW photographs, separate from labelled training images."""
    location=Path(source).expanduser().resolve() if source else Path(cfg.TEST_IMAGES)
    if not location.exists():
        raise FileNotFoundError(f'Test images not found: {location}. Copy photographs into '
                                'Step_02_Training/data/test_images/ or pass --source.')
    files=([location] if location.is_file() else
           sorted(p for p in location.iterdir() if p.suffix.lower() in EXTENSIONS and p.is_file()))
    if not files:
        raise RuntimeError(f'No JPG/PNG photographs in {location}. '
                           'Copy new test images into Step_02_Training/data/test_images/.')
    return files[:cfg.TEST_IMAGE_LIMIT]


def _step1_label_helpers():
    """Reuse exactly the same tiled YOLO and prompted SAM logic as Step 1."""
    import importlib.util
    source=Path(cfg.STEP1_DATA).parent/'main.py'
    if not source.is_file():
        raise RuntimeError(f'Cannot find Step 1 editor code: {source}')
    spec=importlib.util.spec_from_file_location('_speedclimb_step1_helpers',source)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_comparison(image, candidates, sam_polys, out):
    """Write three full-resolution views plus an easy-to-compare contact sheet."""
    from PIL import ImageFont
    out.mkdir(parents=True,exist_ok=True)
    colours={'box':(250,192,48,255),'yolo':(34,202,113,255),'sam':(44,195,248,255)}
    views=[]
    for kind in ('box','yolo','sam'):
        canvas=image.convert('RGBA')
        overlay=Image.new('RGBA',canvas.size,(0,0,0,0))
        draw=ImageDraw.Draw(overlay)
        for item in candidates if kind!='sam' else [{'poly':poly} for poly in sam_polys]:
            if kind=='box':
                x0,y0,x1,y1=item['box'];draw.rectangle((x0,y0,x1,y1),outline=colours[kind],width=max(2,int(max(image.size)/500)))
            else:
                poly=item.get('polygon') if kind=='yolo' else item['poly']
                if poly is not None and len(poly)>=3:
                    draw.polygon(poly,fill=(*colours[kind][:3],45),outline=colours[kind])
        finished=Image.alpha_composite(canvas,overlay).convert('RGB')
        filename={'box':'yolo_boxes.jpg','yolo':'yolo_segmentation.jpg',
                  'sam':'yolo_boxes_plus_sam.jpg'}[kind]
        finished.save(out/filename,quality=93)
        views.append(finished)
    titles=['YOLO bounding boxes','YOLO segmentation','YOLO boxes + SAM']
    max_panel_w=750; max_panel_h=830;head=44
    scale=min(max_panel_w/image.width,max_panel_h/image.height,1.)
    pw=max(1,round(image.width*scale));ph=max(1,round(image.height*scale))
    board=Image.new('RGB',(pw*3,ph+head),(247,248,250))
    drawer=ImageDraw.Draw(board)
    for i,(view,title) in enumerate(zip(views,titles)):
        resized=view.resize((pw,ph),Image.Resampling.LANCZOS)
        board.paste(resized,(pw*i,head))
        drawer.text((pw*i+12,13),title,fill=(25,33,44))
    board.save(out/'comparison.jpg',quality=94)


def test(source=None, evaluate_held_out=False, weights=None):
    """Compare the actual YOLO masks and YOLO boxes prompted through SAM 2."""
    files=comparison_paths(source)
    modelpath=Path(weights).expanduser().resolve() if weights else latest_model()
    if modelpath is None:
        _,_,step1_weights=paths()
        if not step1_weights.is_file():
            raise RuntimeError('No trained model yet. Train first, or put a YOLO checkpoint '
                               'at Step_01_Labeling/data/best.pt.')
        modelpath=step1_weights
        print('No Step 2 checkpoint. Comparing with existing Step 1 best.pt instead.')
    try:from ultralytics import YOLO,SAM
    except ImportError as e:raise RuntimeError('Install Ultralytics: python -m pip install ultralytics') from e
    model=YOLO(str(modelpath))
    sam_file=Path(cfg.STEP1_DATA)/'sam2_t.pt'
    sam=SAM(str(sam_file) if sam_file.is_file() else 'sam2_t.pt')
    helpers=_step1_label_helpers()
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    destination=Path(cfg.RESULTS)/'predictions'/f'comparison_{stamp}'
    destination.mkdir(parents=True,exist_ok=True)
    records=[]
    print(f'YOLO: {modelpath} ({model.task})\nTest photographs: {len(files)}\nSAM: sam2_t.pt')
    if evaluate_held_out:
        dataset=Path(cfg.RESULTS)/'dataset'
        split_dir=dataset/'images'/'test'
        if split_dir.is_dir() and list(split_dir.glob('*.jpg')):
            print('Running quantitative evaluation on the reserved TEST split ...')
            metrics=model.val(data=str(dataset/'data.yaml'),split='test',device=device(),
                              imgsz=cfg.IMAGE_SIZE,project=str(destination),name='metrics',plots=True)
            print(f'Box mAP50: {metrics.box.map50:.3f}')
            if model.task=='segment':print(f'Mask mAP50: {metrics.seg.map50:.3f}')
        else:print('No reserved held-out test split. Skipping quantitative metrics.')
    for i,path in enumerate(files,1):
        with Image.open(path) as picture:
            rgb=ImageOps.exif_transpose(picture).convert('RGB')
        candidates=helpers.detect_yolo_candidates(rgb,model,confidence=cfg.TEST_CONFIDENCE,
                                                    tile_size=cfg.TEST_TILE_SIZE,
                                                    overlap=cfg.TEST_TILE_OVERLAP)
        sam_per_box=[helpers.sam_mask_for_box(rgb,sam,item['box']) for item in candidates]
        sam_polys=[poly for poly in sam_per_box if poly is not None]
        folder=destination/f'{i:03d}_{path.stem}'
        render_comparison(rgb,candidates,sam_polys,folder)
        # Step 3 can consume either set of coordinates without re-running inference.
        predicted=dict(image=str(path),image_size_wh=list(rgb.size),
                       model=str(modelpath),detections=[
                           dict(box_xyxy_px=item['box'],confidence=item['score'],
                                yolo_polygon_px=item['polygon'],sam_polygon_px=sam_per_box[j])
                           for j,item in enumerate(candidates)])
        (folder/'predictions.json').write_text(json.dumps(predicted,indent=2),encoding='utf8')
        summary=dict(image=str(path),model=str(modelpath),task=model.task,
                     yolo_boxes=len(candidates),yolo_segmentation_masks=sum(
                         c['polygon'] is not None for c in candidates),sam_masks=len(sam_polys))
        (folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
        records.append(summary)
        print(f'[{i}/{len(files)}] {path.name}: {len(candidates)} YOLO boxes; '
              f'{summary["yolo_segmentation_masks"]} YOLO masks; {len(sam_polys)} SAM masks')
    (destination/'summary.json').write_text(json.dumps(records,indent=2),encoding='utf8')
    print(f'\nComparison images saved to: {destination}')
    if model.task!='segment':
        print('NOTE: The current checkpoint is detection-only, so the YOLO segmentation view '
              'will be blank. Train YOLO26n-seg to compare both masks.')
    return len(records)


def install_model(source,destination):
    """Copy already validated checkpoint with backup and atomic replacement."""
    source=Path(source);destination=Path(destination)
    if not source.is_file():raise FileNotFoundError(source)
    if source.resolve()==destination.resolve():raise ValueError('Source and destination must differ.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    backup=None
    if destination.exists():
        backups=Path(cfg.RESULTS)/'backups';backups.mkdir(parents=True,exist_ok=True)
        now=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        backup=backups/f'best_before_update_{now}.pt'
        shutil.copy2(destination,backup)
    with tempfile.NamedTemporaryFile(prefix='.best_new_',suffix='.pt',dir=destination.parent,delete=False) as tmp:
        temporary=Path(tmp.name)
    try:
        shutil.copy2(source,temporary)
        if temporary.stat().st_size!=source.stat().st_size:
            raise RuntimeError('Incomplete model copy; old model left untouched.')
        os.replace(temporary,destination)
    finally:
        if temporary.exists():temporary.unlink()
    return backup


def update(assume_yes=False):
    trained=latest_model()
    if trained is None:raise RuntimeError('No training checkpoint found in results/training/.../weights/best.pt')
    _,_,destination=paths()
    # Validate checkpoint before overwriting a working model.
    try:from ultralytics import YOLO
    except ImportError as e:raise RuntimeError('Ultralytics is required to validate this model.') from e
    loaded=YOLO(str(trained))
    if loaded.task not in ('segment','detect'):
        raise RuntimeError(f'New checkpoint is neither detection nor segmentation: {loaded.task}')
    if len(loaded.names)!=1:
        raise RuntimeError(f'Expected one speed-hold class, found {len(loaded.names)} classes. '
                           'Step 1 best.pt was not changed.')
    print(f'New model: {trained}\nStep 1 model to replace: {destination}')
    if not assume_yes:
        answer=input('Update Step 1 best.pt? Existing file will be backed up [y/N]: ').strip().lower()
        if answer not in ('y','yes'):
            print('Cancelled. Nothing changed.');return None
    backup=install_model(trained,destination)
    if backup:print(f'Old model backup: {backup}')
    print(f'Step 1 now uses: {destination}\nRestart Step 1 to load the new model.')
    return destination


def menu():
    while True:
        print('\n'+'='*52+'\n  STEP 02 - HOLD MODEL TRAINING\n'+'='*52)
        print('  1. Prepare/check dataset from Step 1')
        print('  2. Train YOLO segmentation model')
        print('  3. Compare YOLO segmentation vs YOLO boxes + SAM')
        print('  4. Update Step 1 best.pt (backup first)')
        print('  0. Exit')
        option=input('\nChoose: ').strip()
        if option=='0':return
        try:
            if option=='1':prepare()
            elif option=='2':
                print('Start from: [1] pretrained YOLO26-seg  [2] existing Step 1 best.pt  [3] previous Step 2 best.pt')
                answer=input('Choice [1]: ').strip()
                train(use_previous=(answer=='3'),from_step1=(answer=='2'))
            elif option=='3':
                folder=input('Test image folder (Enter = data/test_images): ').strip().strip('"')
                test(folder or None)
            elif option=='4':update()
            else:print('Choose 0, 1, 2, 3 or 4.')
        except (OSError,ValueError,RuntimeError) as exc:
            print('\nERROR:',exc)
        except KeyboardInterrupt:
            print('\nCancelled.')


def main():
    parser=argparse.ArgumentParser(description='Step 02. No args opens a four-option menu.')
    parser.add_argument('action',nargs='?',choices=['prepare','train','test','update'])
    parser.add_argument('--source',help='Folder or image for Test; default data/test_images')
    parser.add_argument('--metrics',action='store_true',help='Also evaluate reserved labelled test split')
    parser.add_argument('--weights',help='Specific YOLO checkpoint for testing (optional)')
    parser.add_argument('--continue-training',action='store_true',help='Start training from previous Step 2 best.pt')
    parser.add_argument('--from-step1',action='store_true',help='Fine-tune existing Step 1 best.pt segmentation model')
    parser.add_argument('--yes',action='store_true',help='Confirm update model without a prompt')
    args=parser.parse_args()
    try:
        if not args.action:menu()
        elif args.action=='prepare':prepare()
        elif args.action=='train':train(args.continue_training,args.from_step1)
        elif args.action=='test':test(args.source,args.metrics,args.weights)
        elif args.action=='update':update(args.yes)
    except (OSError,ValueError,RuntimeError) as exc:
        parser.exit(1,f'ERROR: {exc}\n')


if __name__=='__main__':main()
