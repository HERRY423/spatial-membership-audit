"""Count-format checks only; never computes labels or spatial endpoints."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from .adapters import tenx_metadata,iter_tenx_cells,iter_cosmx_cells

def inspect(path,platform):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    rows=0;nnz=0
    if platform=='xenium':
        meta=tenx_metadata(path)
        for a,b,c in iter_tenx_cells(path):rows+=b-a;nnz+=int(np.count_nonzero(c.data))
        features=int(meta['shape'][0]);genes=int(np.sum(meta['feature_types']=='Gene Expression'))
    else:
        seen=set()
        for c,f in iter_cosmx_cells(path):
            keys=list(zip(c.fov,c.cell_ID))
            if len(keys)!=len(set(keys)) or seen.intersection(keys):raise ValueError('Duplicate CosMx fov/cell_ID')
            seen.update(keys);rows+=len(c);nnz+=int(np.count_nonzero(c[f['count_names']].to_numpy()))
        if not rows:raise ValueError('Empty CosMx matrix')
        features=len(f['count_names']);genes=len(f['genes'])
    return dict(status='COUNT_FORMAT_PASS',sha256=h.hexdigest(),cells=rows,features=features,gene_features=genes,nonzero_entries=nnz,identity_run=False,spatial_run=False,coordinate_units_validated=False)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--platform',choices=['cosmx','xenium'],required=True);p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out)
    if out.exists():p.error('Output already exists')
    result=inspect(a.input,a.platform)
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))

if __name__=='__main__':main()
