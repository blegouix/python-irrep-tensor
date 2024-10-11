import numpy as np
import itertools
from vendor.pysyt.pysyt.syt import * 

from fill import *

# Young tableau
class YoungTableau:
    m_tableau = [[]] 
    m_shape = np.empty(0)
    m_r = 0
    m_d = 0

    def __init__(self, tableau, d):
        self.m_tableau = tableau
        self.m_shape = np.array([len(row) for row in tableau])
        self.m_r = np.sum(self.m_shape)
        self.m_d = d 

    def irrep_dim(self):
        hooks = hook_lengths(self.m_shape)
        prod = 1 
        for i in range(0, len(hooks)):
            row = hooks[i]
            for j in range(0, len(row)):
                prod = prod * (self.m_d+j-i)/row[j]
        return int(prod)

    def projector(self):
        def tr_lambda(idx_to_permute):
            def lambda_to_return(T, idx):
                if all([idx[i]==idx[idx_to_permute[i]+self.m_r] for i in range(0, len(idx_to_permute))]):
                    T[idx] = 1
            return lambda_to_return 
        I = np.zeros(tuple([self.m_d for i in range(0, 2*self.m_r)]))
        fill(I, (), tr_lambda([i for i in range(0, self.m_r)]))
        proj = np.copy(I)

        def permutation_parity(lst):
            '''\
            Given a permutation of the digits 0..N in order as a list, 
            returns its parity (or sign): +1 for even parity; -1 for odd.
            '''
            parity = 1
            for i in range(0,len(lst)-1):
                if lst[i] != i:
                    parity *= -1
                    mn = min(range(i,len(lst)), key=lst.__getitem__)
                    lst[i],lst[mn] = lst[mn],lst[i]
            return parity

        def permutations_subset(t, subset_values):
            '''\
            Compute all permutations on a subset of indexes in t, keep the
            rest of the indexes where they are
            '''
            subset_indexes = [i for i, x in enumerate(t) if x in subset_values]

            elements_to_permute = [t[i] for i in subset_indexes]
            
            perms = itertools.permutations(elements_to_permute)
            
            result = []
            
            for perm in perms:
                temp = list(t)
                for idx, val in zip(subset_indexes, perm):
                    temp[idx] = val
                result.append(tuple(temp))
            
            return result

        for row in self.m_tableau:
            if len(row)>=2:
                row_permutations = permutations_subset(tuple(range(1, self.m_r+1)), row)
                Sym = np.zeros(tuple([self.m_d for j in range(0, 2*self.m_r)]))
                for row_permutation in row_permutations:
                    Tr = np.zeros(tuple([self.m_d for j in range(0, 2*self.m_r)]))
                    fill(Tr, (), tr_lambda([elem-1 for elem in row_permutation]))
                    Sym = Sym+Tr/len(row_permutations)
                proj = np.tensordot(Sym, proj, axes=self.m_r)
        tableau_dual = dual_syt(self.m_tableau)
        for row in tableau_dual:
            if len(row)>=2:
                row_permutations = permutations_subset(tuple(range(1, self.m_r+1)), row)
                AntiSym = np.zeros(tuple([self.m_d for j in range(0, 2*self.m_r)]))
                for row_permutation in row_permutations:
                    Tr = np.zeros(tuple([self.m_d for j in range(0, 2*self.m_r)]))
                    fill(Tr, (), tr_lambda([elem-1 for elem in row_permutation]))
                    AntiSym = AntiSym+permutation_parity([elem-1 for elem in row_permutation])*Tr/len(row_permutations)
                proj = np.tensordot(AntiSym, proj, axes=self.m_r)
        return proj
