"""Capture a failed numerical label comparison before raising an error."""
from pathlib import Path
import hashlib,json,uuid,platform,sys
import numpy as np

def assert_label_match(actual, expected, cell_ids, output_dir, *, context, source_paths=()):
    a,b=np.asarray(actual),np.asarray(expected)
    if np.array_equal(a,b):return
    # Do not truncate the failed arrays: shapes, values and keys are needed to diagnose it.
    ids=np.asarray(cell_ids,dtype=str)
    dest=Path(output_dir)/('label_failure_'+uuid.uuid4().hex)
    dest.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(dest/'failed_arrays.npz',actual=a,expected=b,cell_ids=ids)
    hashes={}
    for path in source_paths:
        p=Path(path);h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1048576),b''):h.update(block)
        hashes[str(p)]=h.hexdigest()
    bad=np.flatnonzero(a!=b).tolist() if a.shape==b.shape else None
    record=dict(context=context,actual_shape=list(a.shape),expected_shape=list(b.shape),
        actual_dtype=str(a.dtype),expected_dtype=str(b.dtype),mismatch_indices=bad,
        python=sys.version,numpy=np.__version__,platform=platform.platform(),source_hashes=hashes)
    (dest/'failure.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    raise AssertionError('Label comparison failed; full snapshot: '+str(dest))
