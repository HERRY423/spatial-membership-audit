"""Panel-aware candidate definition v1. Synthetic tests only; not identity truth."""
import numpy as np
from .adapters import validate_counts

RULE_ID='common125_candidate_v1'
REQUIRED=('MS4A1','CD19','CD79A','CD3D','CD3E','CD8A','CD4','CD14','CD68','MZB1','TNFRSF17')

def panel_contract(genes):
    if len(genes)!=len(set(genes)):raise ValueError('Duplicate gene names')
    missing=sorted(set(REQUIRED)-set(genes))
    return dict(rule_id=RULE_ID,required_genes=list(REQUIRED),missing_genes=missing,
                status='NOT_IDENTIFIABLE' if missing else 'CANDIDATE_ONLY_IDENTITY_NOT_VALIDATED')

def candidate_labels(counts,genes,baseline_eligible):
    """Frozen baseline eligibility supplied by caller; never reselect QC after thinning.

    All rules require measured features. Missing panel features are not zero-filled.
    This is a proposed operational classifier; real-data execution needs phase B approval.
    """
    c=np.asarray(counts);validate_counts(c)
    if c.ndim!=2 or c.shape[1]!=len(genes):raise ValueError('Counts/genes shape mismatch')
    mask=np.asarray(baseline_eligible)
    if mask.dtype!=bool or mask.shape!=(len(c),):raise ValueError('Explicit baseline boolean mask required')
    contract=panel_contract(genes)
    if contract['missing_genes']:raise ValueError('NOT_IDENTIFIABLE: '+','.join(contract['missing_genes']))
    g={x:c[:,genes.index(x)] for x in REQUIRED}
    b=(g['MS4A1']>=1)&((g['CD19']>=1)|(g['CD79A']>=1))
    t=(g['CD3D']>=1)&(g['CD3E']>=1)&((g['CD3D']+g['CD3E'])>=3)
    m=(g['CD14']>=1)&(g['CD68']>=1)
    plasma=(g['MZB1']>=1)&(g['TNFRSF17']>=1)
    conflict=(b.astype(int)+t.astype(int)+m.astype(int)>1)|((b|t|m)&plasma)
    labels=np.full(len(c),'UNRESOLVED',dtype=object)
    labels[plasma]='PLASMA_LIKE_EXCLUDED'
    labels[b&~plasma]='B_CANDIDATE'
    labels[m&~plasma]='MYELOID_CANDIDATE'
    cd8=t&(g['CD8A']>=2);cd4=t&(g['CD4']>=1)
    labels[cd8&~cd4]='CD8_T_CANDIDATE'
    labels[cd4&~cd8]='CD4_T_CANDIDATE'
    labels[t&~cd8&~cd4]='T_UNRESOLVED'
    labels[conflict|(cd8&cd4)]='AMBIGUOUS'
    labels[~mask]='LOW_DETECTION_BASELINE'
    return labels
