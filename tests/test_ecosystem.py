import itertools
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy import sparse
from spatial_membership_audit.ecosystem import (
    SpatialGraph, build_graph, random_label_moments, weighted_pair_moments,
    cross_k, ripley_weights, align_annotation, from_celltypist, categorical_input,
    giotto_endpoint_null_moments, predict_cross_k,graph_from_anndata,giotto_edge_table)


class EcosystemTests(unittest.TestCase):
    def test_giotto_stub_null_full_enumeration(self):
        g=sparse.csr_matrix([[0,1,1,1],[1,0,0,0],[1,0,0,0],[1,0,0,0]])
        labels=np.array(['A','B','B','C'])
        out=giotto_endpoint_null_moments(g,labels,('A','B'))
        stubs=['A','A','A','B','B','C']; values=[]
        for seq in set(itertools.permutations(stubs)):
            values.append(sum({seq[i],seq[i+3]}=={'A','B'} for i in range(3)))
        self.assertAlmostEqual(out['mean'],np.mean(values))
        self.assertAlmostEqual(out['variance'],np.var(values))
        self.assertNotAlmostEqual(out['mean'],random_label_moments(g,1,2)['mean'])

    def test_k_propagation_deterministic_and_zero(self):
        xy=[[1,1],[2,1],[8,8]]; p=np.array([[1,0],[0,1],[0,1]])
        result=predict_cross_k(p,xy,[.5,1.1],window=[0,10,0,10],correction='translation',independent_nodes=True)
        expected=cross_k(xy,['A','B','B'],('A','B'),[.5,1.1],window=[0,10,0,10],correction='translation')
        np.testing.assert_allclose([r['K_at_conditional_means'] for r in result],[r['K'] for r in expected])
        self.assertEqual(result[1]['var_K_delta'],0)
        self.assertIsNone(result[0]['var_L_delta'])

    def test_random_label_exact_enumeration(self):
        rng=np.random.default_rng(42)
        for n in range(2,7):
            upper=np.triu(rng.uniform(0,2,(n,n))*(rng.random((n,n))>.4),1)
            g=sparse.csr_matrix(upper+upper.T)
            for a,b in [(1,1),(1,n-1),(0,n)]:
                vals=[]
                for ia in itertools.combinations(range(n),a):
                    rest=sorted(set(range(n))-set(ia))
                    for ib in itertools.combinations(rest,b):
                        aa=np.isin(np.arange(n),ia).astype(float)
                        bb=np.isin(np.arange(n),ib).astype(float)
                        vals.append(float(aa@g@bb))
                out=random_label_moments(g,a,b)
                self.assertAlmostEqual(out['mean'],np.mean(vals),places=10)
                self.assertAlmostEqual(out['variance'],np.var(vals),places=10)

    def test_same_E_different_variance(self):
        # Both graphs have four edges, N=5 and A=B=1: same E, but use A=1 B=2
        def graph(edges):
            m=np.zeros((5,5))
            for i,j in edges: m[i,j]=m[j,i]=1
            return sparse.csr_matrix(m)
        star=graph([(0,1),(0,2),(0,3),(0,4)])
        path=graph([(0,1),(1,2),(2,3),(3,4)])
        a=random_label_moments(star,1,2); b=random_label_moments(path,1,2)
        self.assertAlmostEqual(a['mean'],b['mean'])
        self.assertNotAlmostEqual(a['variance'],b['variance'])

    def test_complete_graph_degenerate(self):
        g=sparse.csr_matrix(np.ones((5,5))-np.eye(5))
        out=random_label_moments(g,1,2)
        self.assertAlmostEqual(out['mean'],2)
        self.assertEqual(out['variance'],0)

    def test_weighted_categorical_full_enumeration(self):
        p=np.array([[.2,.4],[.5,.2],[.3,.1],[.1,.6]])
        g=sparse.csr_matrix([[0,.4,0,1.1],[.4,0,.7,0],[0,.7,0,.2],[1.1,0,.2,0]])
        prob=[]; values=[]
        for states in itertools.product(range(3),repeat=4):
            mass=np.prod([p[i,s] if s<2 else 1-p[i].sum() for i,s in enumerate(states)])
            a=(np.array(states)==0).astype(float); b=(np.array(states)==1).astype(float)
            prob.append(mass); values.append([float(a@g@b),a.sum()*b.sum()])
        prob=np.array(prob); values=np.array(values); mu=prob@values
        covariance=(values-mu).T@(prob[:,None]*(values-mu))
        out=weighted_pair_moments(p,g,independent_nodes=True)
        np.testing.assert_allclose([out['mu_T'],out['mu_D']],mu,atol=1e-12)
        np.testing.assert_allclose([[out['var_T'],out['cov_T_D']],[out['cov_T_D'],out['var_D']]],covariance,atol=1e-12)
        self.assertAlmostEqual(out['p_D_positive'],prob[values[:,1]>0].sum())

    def test_weighted_matches_binary_existing(self):
        from spatial_membership_audit.analytic import pair_moments
        p=np.array([[.1,.4],[.2,.2],[.3,.5]])
        g=sparse.csr_matrix([[0,1,0],[1,0,1],[0,1,0]])
        out=weighted_pair_moments(p,g,independent_nodes=True); prior=pair_moments(p,g)
        c=2*2/(3*2)
        np.testing.assert_allclose([out['mu_T'],out['var_T'],out['cov_T_D']*c],
                                  [prior['mu_O'],prior['var_O'],prior['cov_O_E']])

    def test_weighted_independence_gate_and_empty(self):
        g=sparse.csr_matrix((2,2)); p=np.array([[1.,0],[0,1.]])
        with self.assertRaises(ValueError): weighted_pair_moments(p,g)
        out=weighted_pair_moments(p,g,independent_nodes=True)
        self.assertEqual(out['ratio_of_conditional_means'],0)
        out=weighted_pair_moments(np.zeros((2,2)),g,independent_nodes=True)
        self.assertIsNone(out['ratio_of_conditional_means'])

    def test_graph_region_and_symmetry_validation(self):
        g=[[0,1],[1,0]]
        with self.assertRaises(ValueError): SpatialGraph(('a','b'),('x','y'),g,False,'supplied')
        with self.assertRaises(ValueError): SpatialGraph(('a','b'),('x','x'),[[0,1],[0,0]],False,'supplied')
        with self.assertRaises(ValueError): SpatialGraph(('a','a'),('x','x'),g,False,'supplied')
        with self.assertRaises(ValueError): SpatialGraph(('a','b'),('x','x'),g,False,'supplied','pixel')

    def test_three_graphs(self):
        xy=[[0,0],[1,0],[.2,1],[2,2],[0,0],[1,0],[.2,1],[2,2]]
        ids=list('abcdefgh'); regions=['x']*4+['y']*4
        for method,kw in [('radius',{'radius_um':1.5}),('knn',{'k':1,'knn_symmetry':'union'}),('delaunay',{})]:
            g=build_graph(xy,ids,regions,method=method,**kw)
            self.assertEqual(g.adjacency[:4,4:].nnz,0)
            self.assertEqual((g.adjacency!=g.adjacency.T).nnz,0)
        with self.assertRaises(ValueError): build_graph(xy,ids,regions,method='knn',k=1)

    def test_anndata_and_giotto_interchange(self):
        import anndata as ad
        obj=ad.AnnData(X=np.zeros((3,1)),obs=pd.DataFrame({'region':['r']*3},index=['a','b','c']))
        obj.obsp['spatial_connectivities']=sparse.csr_matrix([[0,1,0],[1,0,1],[0,1,0]])
        graph=graph_from_anndata(obj,region_key='region',directed=False,coordinate_unit='um')
        edges=giotto_edge_table(graph,[[0,0],[3,4],[3,6]])
        self.assertEqual(list(edges.distance),[5,2])
        self.assertEqual(list(edges['from']),['a','b'])
        self.assertEqual(set(edges.columns),{'from','to','sdimx_begin','sdimy_begin','sdimx_end','sdimy_end','distance','weight'})
        self.assertEqual(obj.obsp['spatial_connectivities'].nnz,4)

    def test_k_manual_and_missing(self):
        xy=[[1,1],[2,1],[8,8]]; labels=['A','B','B']; window=[0,10,0,10]
        row=cross_k(xy,labels,('A','B'),[1.1],window=window,correction='none')[0]
        self.assertEqual(row['K'],50.)
        row=cross_k(xy,labels,('A','B'),[1.1],window=window,correction='translation')[0]
        self.assertAlmostEqual(row['K'],100*(100/90)/2)
        absent=cross_k(xy,labels,('A','C'),[1.1],window=window,correction='none')[0]
        self.assertIsNone(absent['K'])
        zero=cross_k(xy,labels,('A','B'),[.5],window=window,correction='none')[0]
        self.assertEqual(zero['K'],0)
        self.assertEqual(zero['L_minus_r'],-.5)
        with self.assertRaises(ValueError): ripley_weights(xy,1,window=[0,1,0,1],correction='none')

    def test_annotation_ids_and_unknown(self):
        table=pd.DataFrame({'id':['c','b','a'],'label':['B',None,'A'],'score':[.5,.1,.8]})
        out=align_annotation(table,['a','b','c'],id_column='id',label_column='label',method='SingleR')
        self.assertEqual(list(out['labels']),['A','UNKNOWN','B'])
        with self.assertRaises(ValueError): align_annotation(table,['a','b'],id_column='id',label_column='label',method='SingleR')

    def test_celltypist_output_contract_only(self):
        # A contract fixture, not a native CellTypist run or model validation.
        native=SimpleNamespace(predicted_labels=pd.DataFrame({'predicted_labels':pd.Categorical(['B','A',None])},index=['b','a','c']))
        self.assertEqual(list(from_celltypist(native,['a','b','c'])['labels']),['A','B','UNKNOWN'])

    def test_probability_semantics(self):
        table=pd.DataFrame({'id':['b','a'],'A':[.2,.7],'B':[.8,.3]})
        kw=dict(id_column='id',class_columns=['A','B'],semantics='categorical_posterior',independent_nodes=True,provenance='synthetic')
        result=categorical_input(table,['a','b'],**kw)
        np.testing.assert_allclose(result['probabilities'],[[.7,.3],[.2,.8]])
        for changed in [{'semantics':'SingleR_scores'},{'independent_nodes':False}]:
            with self.assertRaises(ValueError): categorical_input(table,['a','b'],**(kw|changed))
        table['B']=.9
        with self.assertRaises(ValueError): categorical_input(table,['a','b'],**kw)


if __name__=='__main__': unittest.main()
