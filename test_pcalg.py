# -*- coding: utf-8 -*-
'''
Test suite for pcalg
'''
import datetime

import networkx as nx
import numpy as np

from gsq.ci_tests import ci_test_bin, ci_test_dis
from gsq.gsq_testdata import bin_data, dis_data

import pytest

from pcalg import estimate_cpdag
from pcalg import estimate_skeleton
from pcalg import estimate_skeleton_parallel

@pytest.mark.parametrize(('indep_test_func', 'data_matrix', 'g_answer'), [
    (ci_test_bin, np.array(bin_data).reshape((5000, 5)), nx.DiGraph({
        0: (1, ),
        1: (),
        2: (3, 4),
        3: (1, 2),
        4: (1, 2),
    })),
    (ci_test_dis, np.array(dis_data).reshape((10000, 5)), nx.DiGraph({
        0: (2, ),
        1: (2, 3),
        2: (),
        3: (),
        4: (3, ),
    })),
])
def test_estimate_cpdag(indep_test_func, data_matrix, g_answer, alpha=0.01):
    '''
    estimate_cpdag should reveal the answer
    '''
    (graph, sep_set) = estimate_skeleton(indep_test_func=indep_test_func,
                                         data_matrix=data_matrix,
                                         alpha=alpha)
    graph = estimate_cpdag(skel_graph=graph, sep_set=sep_set)
    error_msg = 'True edges should be: %s' % (g_answer.edges(), )
    assert nx.is_isomorphic(graph, g_answer), error_msg

def test_fixed_edges():
    '''
    The fixed edges shall appear in the skeleton
    '''
    data_matrix = np.array(bin_data).reshape((5000, 5))
    (graph, sep_set) = estimate_skeleton(indep_test_func=ci_test_bin,
                                         data_matrix=data_matrix,
                                         alpha=0.01)
    graph = estimate_cpdag(skel_graph=graph, sep_set=sep_set)
    assert not graph.has_edge(1, 2)

    fixed_edges = nx.DiGraph()
    fixed_edges.add_nodes_from(range(5))
    fixed_edges.add_edge(1, 2)
    with pytest.raises(ValueError):
        _ = estimate_skeleton(indep_test_func=ci_test_bin,
                              data_matrix=data_matrix,
                              alpha=0.01,
                              fixed_edges=((1,2), ))
    with pytest.raises(ValueError):
        _ = estimate_skeleton(indep_test_func=ci_test_bin,
                              data_matrix=data_matrix,
                              alpha=0.01,
                              fixed_edges=nx.DiGraph({0: (1, )}))
    (graph, _) = estimate_skeleton(indep_test_func=ci_test_bin,
                                   data_matrix=data_matrix,
                                   alpha=0.01,
                                   fixed_edges=fixed_edges)
    assert graph.has_edge(1, 2), graph.edges

@pytest.mark.parametrize(('indep_test_func', 'data_matrix'), [
    (ci_test_bin, np.array(bin_data).reshape((1000, 25))),
    (ci_test_dis, np.array(dis_data).reshape((2000, 25))),
])
def test_parallel_estimate_skeleton(indep_test_func, data_matrix, alpha=0.01):
    '''
    estimate_skeleton_parallel should be faithful to stable estimate_skeleton
    '''
    start = datetime.datetime.now()
    (graph, sep_set) = estimate_skeleton(indep_test_func=indep_test_func,
                                         data_matrix=data_matrix,
                                         alpha=alpha,
                                         method='stable')
    print('Duration of stable estimate_skeleton:',
          datetime.datetime.now() - start)
    start = datetime.datetime.now()
    result = estimate_skeleton_parallel(indep_test_func=indep_test_func,
                                        data_matrix=data_matrix,
                                        alpha=alpha,
                                        num_cores=4)
    print('Duration of estimate_skeleton_parallel:',
          datetime.datetime.now() - start)

    assert len(result) == 2
    assert nx.is_isomorphic(graph, result[0])
    sep_set_parallel = result[1]
    assert len(sep_set) == len(sep_set_parallel)
    for i, sep_set_i in enumerate(sep_set):
        assert len(sep_set_i) == len(sep_set_parallel[i])
        for j, sep_set_ij in enumerate(sep_set_i):
            assert sep_set_ij == sep_set_parallel[i][j]
