"""Fixed-cell descriptive diagnostics; never collapse missing endpoints to zero."""
import csv,hashlib,json,math,os,platform,tempfile,shutil
from pathlib import Path
import numpy as np
import scipy
from scipy.spatial import cKDTree
from . import __version__

def read_tsv(path,required):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f,delimiter='\t')
        if not set(required)<=set(r.fieldnames or []):raise ValueError(f'Missing columns: {required}')
        rows=list(r)
    if not rows:raise ValueError('Empty input table')
    if any(any(row.get(k) is None or not row[k].strip() for k in required) for row in rows):raise ValueError('Missing required value')
    return rows

def membership(before,after):
    union=before|after
    return dict(baseline_count=len(before),current_count=len(after),retained=len(before&after),lost=len(before-after),gained=len(after-before),jaccard=len(before&after)/len(union) if union else None,membership_status='DEFINED' if union else 'EMPTY_UNION')

def endpoints(xy,labels,a,b,radius,edges=None):
    """Undirected O/E, and directed observed-window nearest-target coverage G_ab(r).

    E conditions on the observed graph and global region counts under uniform
    random labelling. G has no boundary correction: it describes this window.
    """
    if edges is None:edges=cKDTree(xy).query_pairs(radius,output_type='ndarray')
    aa=np.flatnonzero(labels==a);bb=np.flatnonzero(labels==b);n=len(labels);m=len(edges)
    obs=int(np.sum(((labels[edges[:,0]]==a)&(labels[edges[:,1]]==b))|((labels[edges[:,0]]==b)&(labels[edges[:,1]]==a)))) if m else 0
    exp=2*m*len(aa)*len(bb)/(n*(n-1)) if n>1 else 0.0
    oe=obs/exp if exp>0 else None
    flags=[]
    if n<2:flags.append('N_LT_2')
    if not len(aa):flags.append('NO_A')
    if not len(bb):flags.append('NO_B')
    if not m:flags.append('NO_GRAPH_EDGES')
    def g(src,dst):
        if not len(src) or not len(dst):return None
        distance=cKDTree(xy[dst]).query(xy[src],k=1)[0]
        return float(np.mean(distance<=radius))
    return dict(n_cells=n,n_edges=m,n_a=len(aa),n_b=len(bb),O=obs,E=exp,OE=oe,OE_status='DEFINED' if exp>0 else '|'.join(flags),G_a_to_b=g(aa,bb),G_b_to_a=g(bb,aa),G_status='DEFINED' if len(aa) and len(bb) else '|'.join(x for x in flags if x.startswith('NO_A') or x.startswith('NO_B')),window_interpretation='OBSERVED_WINDOW_NO_EDGE_CORRECTION')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def run(cells_path,labels_path,pair_args,radius,out):
    if not math.isfinite(radius) or radius<=0:raise ValueError('radius must be finite and positive')
    target=Path(out).resolve()
    if target.exists():raise FileExistsError('Refusing to overwrite existing output')
    pairs=[]
    for spec in pair_args:
        parts=spec.split(',')
        if len(parts)!=2 or any(not x.strip() for x in parts) or parts[0]==parts[1]:raise ValueError('Each pair requires two distinct nonempty labels')
        pair=tuple(parts)
        if pair in pairs or pair[::-1] in pairs:raise ValueError('Duplicate undirected pair')
        pairs.append(pair)
    if not pairs:raise ValueError('At least one pair required')
    cells=read_tsv(cells_path,['cell_id','region_id','x_um','y_um'])
    ids=[c['cell_id'] for c in cells]
    if len(set(ids))!=len(ids):raise ValueError('cell_id must be globally unique')
    xy=np.array([[float(c['x_um']),float(c['y_um'])] for c in cells])
    if not np.isfinite(xy).all():raise ValueError('Coordinates must be finite')
    records=read_tsv(labels_path,['cell_id','replicate','label'])
    trajectories={};known=set(ids)
    for row in records:
        rep=row['replicate'];key=row['cell_id']
        if key not in known:raise ValueError('Labels contain a cell absent from cells table')
        d=trajectories.setdefault(rep,{})
        if key in d:raise ValueError('Duplicate cell/replicate label')
        d[key]=row['label']
    if 'baseline' not in trajectories:raise ValueError('replicate=baseline is required')
    if any(set(d)!=known for d in trajectories.values()):raise ValueError('Every replicate must cover the identical cell universe; use explicit UNKNOWN or LOW_DETECTION labels')
    labels={r:np.array([d[i] for i in ids]) for r,d in trajectories.items()}
    region=np.array([c['region_id'] for c in cells]);id_array=np.array(ids);result=[]
    for reg in sorted(set(region)):
        ix=np.flatnonzero(region==reg);points=xy[ix];edges=cKDTree(points).query_pairs(radius,output_type='ndarray')
        for a,b in pairs:
            base=labels['baseline'][ix];base_e=endpoints(points,base,a,b,radius,edges)
            for rep in ['baseline']+sorted(set(labels)-{'baseline'}):
                current=labels[rep][ix];row=dict(region_id=reg,replicate=rep,label_a=a,label_b=b,radius_um=radius)
                row.update(endpoints(points,current,a,b,radius,edges))
                for name in [a,b]:
                    mem=membership(set(id_array[ix][base==name]),set(id_array[ix][current==name]));prefix='a_' if name==a else 'b_'
                    row.update({prefix+k:v for k,v in mem.items()})
                for metric in ['OE','G_a_to_b','G_b_to_a']:
                    ref=base_e[metric];cur=row[metric];finite=ref is not None and cur is not None
                    row['baseline_'+metric]=ref
                    row['delta_'+metric]=cur-ref if finite else None
                    row['paired_'+metric+'_status']='FINITE_ZERO_CHANGE' if finite and cur==ref else ('FINITE_NONZERO_CHANGE' if finite else 'UNESTIMABLE_PAIR')
                result.append(row)
    receipt=dict(version=__version__,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,inputs={k:dict(name=Path(v).name,sha256=digest(v)) for k,v in [('cells',cells_path),('labels',labels_path)]},radius_um=radius,pairs=pairs,rows=len(result),replicates=sorted(labels),inference='DESCRIPTIVE_ONLY',donor_inference=False,labels_are_ground_truth=False,graph='within-region Euclidean undirected fixed nodes',G_edge_correction='none; observed window only',missing_values='NA',scientific_validation=False)
    target.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='audit-stage-',dir=target.parent))
    try:
        with (stage/'endpoints.tsv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=result[0],delimiter='\t');w.writeheader();w.writerows({k:'NA' if v is None else v for k,v in row.items()} for row in result)
        (stage/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False),encoding='utf-8')
        os.rename(stage,target)
    except BaseException:
        # Only the uniquely created staging directory inside the requested parent.
        if stage.exists() and stage.resolve().parent==target.parent.resolve() and stage.name.startswith('audit-stage-'):shutil.rmtree(stage)
        raise
    return result
