"""Read-only, bounded count adapters. No annotation, graph or model execution."""
import csv,gzip
import numpy as np
from scipy import sparse

def validate_counts(data):
    a=np.asarray(data)
    if not np.issubdtype(a.dtype,np.number) or not np.isfinite(a).all() or (a<0).any() or not np.equal(a,np.floor(a)).all():
        raise ValueError('Counts must be finite nonnegative integers; normalized values are not accepted')
    if a.size and int(a.max())>np.iinfo(np.int64).max:raise ValueError('Count exceeds int64')

def tenx_metadata(path):
    import h5py
    with h5py.File(path,'r') as f:
        g=f['matrix'];shape=tuple(int(x) for x in g['shape'][:])
        decode=lambda k:np.array([x.decode() if isinstance(x,bytes) else str(x) for x in g[k][:]])
        names=decode('features/name');types=decode('features/feature_type');ids=decode('barcodes')
        if len(shape)!=2 or min(shape)<0 or len(names)!=shape[0] or len(types)!=shape[0] or len(ids)!=shape[1]:raise ValueError('H5 axes mismatch')
        if len(set(ids))!=len(ids):raise ValueError('Duplicate H5 barcode')
        gene=types=='Gene Expression'
        if len(set(names[gene]))!=gene.sum():raise ValueError('Duplicate gene symbols need explicit mapping')
        ptr=g['indptr'][:]
        if not np.issubdtype(ptr.dtype,np.integer) or len(ptr)!=shape[1]+1 or ptr[0]!=0 or ptr[-1]!=len(g['data']) or np.any(ptr[1:]<ptr[:-1]) or len(g['indices'])!=len(g['data']):raise ValueError('Invalid CSC structure')
    return dict(shape=shape,barcodes=ids,names=names,feature_types=types)

def iter_tenx_cells(path,batch_cells=25000):
    """Yield start, stop, cells-by-features CSR while preserving input order."""
    import h5py
    if batch_cells<=0:raise ValueError('batch_cells must be positive')
    tenx_metadata(path)
    with h5py.File(path,'r') as f:
        g=f['matrix'];n_features,n_cells=map(int,g['shape'][:])
        for start in range(0,n_cells,batch_cells):
            stop=min(n_cells,start+batch_cells);ptr=g['indptr'][start:stop+1];lo,hi=int(ptr[0]),int(ptr[-1]);idx=g['indices'][lo:hi];data=g['data'][lo:hi]
            validate_counts(data)
            if not np.issubdtype(idx.dtype,np.integer) or (idx.size and (idx.min()<0 or idx.max()>=n_features)):raise ValueError('Feature index out of range')
            mat=sparse.csr_matrix((data.astype(np.int64),idx,ptr-ptr[0]),shape=(stop-start,n_features))
            if not mat.has_canonical_format:raise ValueError('Repeated or unsorted feature indices per cell')
            yield start,stop,mat

def cosmx_features(columns):
    if len(columns)!=len(set(columns)):raise ValueError('Duplicate column names')
    keys={'fov','cell_ID','cell','cell_id'}
    count_names=[x for x in columns if x not in keys]
    negative=[x for x in count_names if x.startswith('Negative') or x.startswith('NegPrb')]
    system=[x for x in count_names if x.startswith('SystemControl')]
    genes=[x for x in count_names if x not in set(negative+system)]
    return dict(genes=genes,negative=negative,system=system,count_names=count_names)

def iter_cosmx_cells(path,chunksize=5000):
    import pandas as pd
    if chunksize<=0:raise ValueError('chunksize must be positive')
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8-sig',newline='') as f:columns=next(csv.reader(f))
    if not {'fov','cell_ID'}<=set(columns):raise ValueError('CosMx requires explicit fov and cell_ID')
    features=cosmx_features(columns)
    with pd.read_csv(path,chunksize=chunksize) as reader:
        for chunk in reader:
            validate_counts(chunk[features['count_names']].to_numpy())
            if chunk[['fov','cell_ID']].isna().any().any():raise ValueError('Missing CosMx cell key')
            validate_counts(chunk[['fov','cell_ID']].to_numpy())
            yield chunk,features
