"""Copy referenced sprites and metadata, preserving sources and verifying hashes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def bank_files(bank):
    with (bank/'view_matrix.csv').open(newline='',encoding='utf-8-sig') as stream:
        rows=list(csv.DictReader(stream))
    files=set()
    for row in rows:
        path=(bank/row['rgba_relpath']).resolve()
        if not path.is_relative_to(bank.resolve()):
            raise ValueError(f'Escaping sprite path: {path}')
        files.add(path)
    files.update(p.resolve() for p in bank.iterdir() if p.is_file() and p.suffix in ('.csv','.json','.npz'))
    return sorted(files)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--asset-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve()
    if out.exists(): raise FileExistsError(out)
    manifest=json.loads((ROOT/'he_renderer/manifests/paper_assets_manifest_v2.json').read_text())
    calibrated=json.loads((ROOT/'he_renderer/manifests/calibrated_renderer_v2.json').read_text())
    copies={}
    def add_bank(source,destination):
        source=source.resolve()
        for file in bank_files(source): copies[destination/file.relative_to(source)]=file
    for asset in manifest['assets']:
        add_bank(args.asset_root/asset['folder'],Path('assets/native')/asset['folder'])
        for side,relative in asset.get('close_banks',{}).items():
            source=(args.asset_root/relative).resolve()
            add_bank(source,Path('assets/pedestrian_close')/source.name)
            asset['close_banks'][side]='../pedestrian_close/'+source.name
    for asset in calibrated['assets']:
        source=(ROOT/'he_renderer/manifests'/asset['close_bank']).resolve()
        add_bank(source,Path('assets/calibrated_close')/source.name)
        asset['close_bank']='../assets/calibrated_close/'+source.name
    # Source-only snapshot; runtime dependencies and driving-model checkpoints are external.
    for source in (ROOT/'he_renderer').rglob('*'):
        rel=source.relative_to(ROOT/'he_renderer')
        if source.is_file() and source.suffix in ('.py','.cu','.json','.txt','.md') and not set(rel.parts)&{'assets','cache','runtime','artifacts','frozen','__pycache__'}:
            copies[Path('source/he_renderer')/rel]=source
    for source in (ROOT/'driving_models/common').glob('*.py'):
        copies[Path('source/driving_models/common')/source.name]=source
    size=sum(p.stat().st_size for p in copies.values())
    if shutil.disk_usage(out.parent).free < size+1024**3: raise RuntimeError('Insufficient free space')
    out.mkdir()
    records=[]
    for i,(relative,source) in enumerate(sorted(copies.items())):
        target=out/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        expected=digest(source)
        if digest(target)!=expected: raise RuntimeError(f'Hash mismatch: {relative}')
        records.append(dict(path=relative.as_posix(),bytes=target.stat().st_size,sha256=expected))
        if i%2000==0: print('Verified',i,'of',len(copies),flush=True)
    (out/'manifests').mkdir()
    for name,data in [('assets.json',manifest),('calibrated.json',calibrated)]:
        path=out/'manifests'/name
        path.write_text(json.dumps(data,indent=2))
        records.append(dict(path=path.relative_to(out).as_posix(),bytes=path.stat().st_size,sha256=digest(path)))
    (out/'INVENTORY.json').write_text(json.dumps(dict(files=records,bytes=sum(r['bytes'] for r in records),
        asset_bytes=sum(r['bytes'] for r in records if r['path'].startswith('assets/')),
        external_dependencies=['CARLA Python API','Python/CUDA/CuPy runtime','model adapters and checkpoints','scenario/route inputs'],
        status='hash verified; rendering validation required'),indent=2))
    (out/'README.md').write_text('# HE Runtime Asset Package\n\n'
        'Use assets/native as --asset-root, manifests/assets.json as --manifest, and\n'
        'manifests/calibrated.json as --he-calibrated-manifest.\n'
        'Enable --he-pedestrian-gpu explicitly for the optimized candidate.\n\n'
        'All original banks are preserved. This is an asset bundle plus source snapshot,\n'
        'not a standalone Python environment. Install dependencies and model adapters separately.\n'
        'Hull caches can be rebuilt from these RGBA files; initial loading may be slow.\n'
        'INVENTORY.json lists SHA256 and bytes for each payload file.\n')
    print('Built',out,'asset GB',sum(r['bytes'] for r in records if r['path'].startswith('assets/'))/1e9,flush=True)


if __name__=='__main__': main()
