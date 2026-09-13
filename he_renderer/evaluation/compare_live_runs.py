"""Equal-time video review of independent closed-loop conditions."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    paths=[]
    for folder in (args.reference,args.candidate):
        videos=list(folder.rglob('*_all_cameras.mp4'))
        if len(videos)!=1: raise ValueError('Expected one mosaic per run')
        paths.append(videos[0])
    caps=[cv2.VideoCapture(str(p)) for p in paths]
    if not all(c.isOpened() for c in caps): raise RuntimeError('Video open failed')
    width,height=int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH)),int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=caps[0].get(cv2.CAP_PROP_FPS)
    if any((int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)),c.get(cv2.CAP_PROP_FPS))!=(width,height,fps) for c in caps):
        raise ValueError('Mismatched video geometry')
    writer=cv2.VideoWriter(str(args.output/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height*2+60))
    if not writer.isOpened(): raise RuntimeError('Video writer failed')
    count=0
    try:
        while True:
            frames=[c.read() for c in caps]
            if not any(ok for ok,_ in frames): break
            if not all(ok for ok,_ in frames): raise RuntimeError('Unequal video duration')
            panel=np.zeros((height*2+60,width,3),np.uint8)
            panel[30:height+30]=frames[0][1]
            panel[height+60:]=frames[1][1]
            cv2.putText(panel,'Earlier physical CARLA | independent trajectories, NOT pose aligned',(8,21),0,.55,(255,255,255),1)
            cv2.putText(panel,f'HE + GPU pedestrian + depth | frame {count}',(8,height+51),0,.55,(255,255,255),1)
            writer.write(panel)
            if count in (200,240,280,340,400,800): cv2.imwrite(str(args.output/f'frame_{count}.png'),panel)
            count+=1
    finally:
        writer.release()
        for cap in caps: cap.release()
    cap=cv2.VideoCapture(str(args.output/'comparison.mp4'))
    decoded=0
    while cap.read()[0]: decoded+=1
    cap.release()
    if decoded!=count: raise RuntimeError('Output decode count mismatch')
    (args.output/'summary.json').write_text(json.dumps(dict(frames=count,fps=fps,inputs=list(map(str,paths)),
        comparison='Equal-time independent closed loops; no pose-aligned image metrics'),indent=2))
    print('Decoded',count,'comparison frames')


if __name__=='__main__': main()
