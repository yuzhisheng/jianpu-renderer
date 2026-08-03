#!/usr/bin/env python3
"""Build an 8-class pitch/rest YOLO dataset from filtered VLM silver labels."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TOKENS = [*(f"P{i}" for i in range(1, 8)), "R0"]
TOKEN_TO_CLASS = {token: index for index, token in enumerate(TOKENS)}


def align(expected, detected):
    n, m = len(expected), len(detected)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): dp[i][0], back[i][0] = i, "up"
    for j in range(1, m + 1): dp[0][j], back[0][j] = j, "left"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = [
                (dp[i-1][j-1] + (0.0 if expected[i-1] == detected[j-1] else 0.45), "diag"),
                (dp[i-1][j] + 1.0, "up"), (dp[i][j-1] + 1.0, "left"),
            ]
            dp[i][j], back[i][j] = min(choices, key=lambda item: item[0])
    pairs=[]; i,j=n,m
    while i or j:
        move=back[i][j]
        if move=="diag": pairs.append((i-1,j-1)); i-=1; j-=1
        elif move=="up": i-=1
        else: j-=1
    return list(reversed(pairs))


def clusters(boxes):
    result=[]
    for box in sorted(boxes, key=lambda item:item[2]):
        target=next((row for row in result if abs(box[2]-sum(x[2] for x in row)/len(row)) <= max(10,box[4]*0.9)),None)
        if target is None:
            result.append([box])
        else:
            target.append(box)
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--annotations",default="backend/real_annotations")
    ap.add_argument("--output",default="backend/pitch_silver_dataset"); ap.add_argument("--min-coverage",type=float,default=.82)
    ap.add_argument("--min-agreement",type=float,default=.55); args=ap.parse_args()
    root=(ROOT/args.annotations).resolve(); out=(ROOT/args.output).resolve()
    if out.exists(): shutil.rmtree(out)
    manifest=[json.loads(x) for x in (root/'manifest.jsonl').read_text().splitlines()]
    pages=sorted(
        (x for x in manifest if x['split']=='train_silver' and x['kind']=='jianpu'),
        key=lambda item: item['annotation_id'],
    )
    val_ids={x['annotation_id'] for i,x in enumerate(pages) if i%5==0}
    stats=dict(rows_seen=0,rows_kept=0,boxes=0,rejected_quality=0,rejected_alignment=0)
    for page in pages:
        aid=page['annotation_id']; review=json.loads((root/page['review']).read_text())
        vlm=json.loads((root/'local_vlm_pitch_reviews'/f'{aid}.json').read_text())
        vlm_rows={x['source_row']:x for x in vlm['rows']}
        with Image.open(root/page['image']) as im: width,height=im.size
        det=[]
        for line in (root/'labels'/f'{aid}.txt').read_text().splitlines():
            c,cx,cy,w,h=map(float,line.split()); det.append((int(c),cx*width,cy*height,w*width,h*height))
        split='val' if aid in val_ids else 'train'; (out/'images'/split).mkdir(parents=True,exist_ok=True); (out/'labels'/split).mkdir(parents=True,exist_ok=True)
        for row in review['rows']:
            stats['rows_seen']+=1; item=vlm_rows.get(row['row']); voices=item.get('voices',[]) if item else []
            expected=[t for t in (voices[0].get('tokens',[]) if len(voices)==1 else []) if t in TOKEN_TO_CLASS]
            if not item or item.get('content_type')!='score' or len(voices)!=1 or item.get('confidence',0)<.9 or item.get('uncertainties') or '?' in voices[0].get('tokens',[]) or len(expected)<3:
                stats['rejected_quality']+=1; continue
            left,top,right,bottom=row['crop_box']; candidates=[x for x in det if x[0]<8 and left<=x[1]<=right and top<=x[2]<=bottom]
            row_clusters=clusters(candidates)
            if not row_clusters: stats['rejected_alignment']+=1; continue
            boxes=min(row_clusters,key=lambda xs:(abs(len(xs)-len(expected)),-len(xs)))
            boxes=sorted(boxes,key=lambda x:x[1]); detected=[TOKENS[x[0]] for x in boxes]; pairs=align(expected,detected)
            coverage=len(pairs)/max(len(expected),len(boxes),1); agreement=sum(expected[i]==detected[j] for i,j in pairs)/max(len(pairs),1)
            if coverage<args.min_coverage or agreement<args.min_agreement:
                stats['rejected_alignment']+=1; continue
            name=f'{aid}_row_{row["row"]:02d}'; source=root/row['image']; target=out/'images'/split/f'{name}.png'; shutil.copyfile(source,target)
            crop_w,crop_h=right-left,bottom-top; lines=[]
            for i,j in pairs:
                _,cx,cy,bw,bh=boxes[j]; lines.append(f'{TOKEN_TO_CLASS[expected[i]]} {(cx-left)/crop_w:.8f} {(cy-top)/crop_h:.8f} {bw/crop_w:.8f} {bh/crop_h:.8f}')
            (out/'labels'/split/f'{name}.txt').write_text('\n'.join(lines)+'\n'); stats['rows_kept']+=1; stats['boxes']+=len(lines)
    (out/'data.yaml').write_text(f'path: {out}\ntrain: images/train\nval: images/val\n\nnc: 8\nnames: {TOKENS}\n')
    stats['train_images']=len(list((out/'images/train').glob('*'))); stats['val_images']=len(list((out/'images/val').glob('*')))
    (out/'stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(stats,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
