import scipy
import numpy as np
import itertools
from vendor.pysyt.pysyt.syt import * 

# CSR storage
class Csr:
    m_ndims = np.empty(0, dtype=int) 
    m_coalesc_idx = np.empty(0, dtype=int) 
    m_idx = np.empty((0, 0), dtype=int) 
    m_values = np.empty(0, dtype=np.double)

    def __init__(self, ndims):
        self.m_ndims = np.array(ndims, dtype=int) 
        self.m_coalesc_idx = np.zeros(1, dtype=int)
        self.m_idx = np.empty((0, len(ndims)-1), dtype=int)
        self.m_values = np.empty(0, dtype=np.double)

    def copy_construct(self, ndims, coalesc_idx, idx, values):
        self.m_ndims = np.array(ndims) 
        self.m_coalesc_idx = coalesc_idx
        self.m_idx = idx
        self.m_values = np.array(values) 
    
    def append_dense(self, dense, **kwargs):
        idx = np.array(np.nonzero(dense))
        self.m_ndims[kwargs.get('axis', None)] = self.m_ndims[kwargs.get('axis', None)]+1
        self.m_coalesc_idx = np.append(self.m_coalesc_idx, self.m_coalesc_idx[-1]+idx.shape[1])
        self.m_idx = np.append(self.m_idx, np.transpose(idx), axis=0)
        self.m_values = np.append(self.m_values, dense[tuple(idx)])

    """
    def append_csr(self, csr, **kwargs):
        self.m_ndims[kwargs.get('axis', None)] = self.m_ndims[kwargs.get('axis', None)]+1
        
        self.m_coalesc_idx = np.append(self.m_coalesc_idx, self.m_coalesc_idx[-1]+csr.m_coalesc_idx)
        self.m_idx = np.append(self.m_idx, csr.m_idx, axis=0)
        self.m_values = np.append(self.m_values, csr.m_values)
    """

    # /!\ support only coalescent axis
    def get(self, id, **kwargs):
        ndims = np.copy(self.m_ndims)
        ndims[kwargs.get('axis', None)] = 1
        csr = Csr(ndims)
        csr.copy_construct(ndims, np.array([0, self.m_coalesc_idx[id+1]-self.m_coalesc_idx[id]]), np.array([self.m_idx[i,:] for i in range(self.m_coalesc_idx[id], self.m_coalesc_idx[id+1])]), np.array([self.m_values[i] for i in range(self.m_coalesc_idx[id], self.m_coalesc_idx[id+1])]))
        return csr 

    # /!\ untested for non-coalescent axis
    def mult(self, x, **kwargs):
        operate_on = kwargs.get('operate_on', None);
        axis_left = list(np.arange(0, len(x.shape)) if operate_on=="left" else np.arange(0, len(self.m_ndims)-len(x.shape))) # in practice this is just list(0)
        # TODO: assert axis_left rank being one
        axis_right = [ax for ax in list(np.arange(0, len(self.m_ndims))) if ax not in axis_left]
        axis_prod = axis_left if operate_on=="left" else axis_right
        axis_ortho = axis_right if operate_on=="left" else axis_left
        prod = np.zeros(self.m_ndims[axis_ortho])
        for coalesc_id_id in range(0, len(self.m_coalesc_idx)-1):
            id_left = (coalesc_id_id,);
            for k in range(self.m_coalesc_idx[coalesc_id_id], self.m_coalesc_idx[coalesc_id_id+1]):
                id_right = tuple(self.m_idx[k,:].astype(int));
                id_prod = id_left if operate_on=="left" else id_right
                id_ortho = id_right if operate_on=="left" else id_left
                prod[id_ortho] = prod[id_ortho] + x[id_prod]*self.m_values[k]
        return prod
                

# csr to dense converter
def csr2dense(csr):
    dense = np.zeros(csr.m_ndims);
    for coalesc_id_id in range(0, len(csr.m_coalesc_idx)-1):
        for i in range(csr.m_coalesc_idx[coalesc_id_id], csr.m_coalesc_idx[coalesc_id_id+1]):
            dense[(csr.m_coalesc_idx[coalesc_id_id],) + tuple([int(csr.m_idx[i, j]) for j in range(0,len(csr.m_ndims)-1)])] = csr.m_values[i]
    return dense

# np.array-represented tensor filler
def fill(T, idx, lambda_func):
    r = len(T.shape)
    d = T.shape[0]
    for i in range(0,d):
        new_idx = idx + (i,)
        if len(new_idx)==r:
            lambda_func(T, new_idx)
        else:
            fill(T, new_idx, lambda_func)


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


# General Tensor class with internal symmetries
class Tensor:
    m_r = 0
    m_d = 0
    m_irrep_dim = 0
    m_U = Csr((0,)) 
    m_V = Csr((0,)) 
    m_data = np.empty(0) 

    # Gram-shmidt approach to produce from vec an orthogonal vector to the vector space generated by basis
    def orthogonalize(self, vec, basis):
        for i in range(0,basis.m_ndims[0]):
            eigentensor = csr2dense(basis.get(i, axis=0)).reshape(tuple([self.m_d for i in range(0, self.m_r)]))
            vec = vec - np.tensordot(vec, eigentensor, axes=self.m_r)/np.tensordot(eigentensor, eigentensor, axes=self.m_r)*eigentensor
        return vec

    # Build orthonormal basis for the eigen subspace associated to the eigenvalue 1 of the projection operator
    def orthonormal_basis_subspace_eigenvalue_1(self, Proj):
        U = Csr(((0,)+tuple([self.m_d for i in range(0, self.m_r)]))); 
        V = Csr(((0,)+tuple([self.m_d for i in range(0, self.m_r)]))); 

        index = 0;
        # while V.m_ndims[0]<scipy.special.binom(self.m_d, self.m_r): # TODO: general formula
        while V.m_ndims[0]<self.m_irrep_dim: # TODO: general formula
            index = index+1;
            def candidate_lambda(T, idx):
                T[idx] = index//2**(np.sum([self.m_d**i*idx[i] for i in range(0,len(idx))]))%2
            candidate = np.zeros(tuple([self.m_d for i in range(0, self.m_r)]))
            fill(candidate, (), candidate_lambda)
            candidate = np.tensordot(Proj, np.random.randint(2, size=tuple([self.m_d for i in range(0, self.m_r)])), axes=self.m_r);
            v = self.orthogonalize(np.tensordot(Proj, candidate, axes=self.m_r), V);

            if np.any(v>0.25):
                v = v/np.sqrt(np.tensordot(v, v, axes=self.m_r)) # normalize
                # u = np.tensordot(np.tensordot(Tr, Proj, axis=self.n_r), v, axis=self.n_r) # not sure if this or u = v is correct
                u = v
                U.append_dense(u.reshape(tuple([self.m_d for i in range(0, self.m_r)])), axis=0)
                V.append_dense(v.reshape(tuple([self.m_d for i in range(0, self.m_r)])), axis=0)
        return [U, V] 

    def __init__(self, young_tableau):
        # TODO: assert Proj hypercubic and pair rank
        self.m_r = young_tableau.m_r 
        self.m_d = young_tableau.m_d
        self.m_irrep_dim = young_tableau.irrep_dim()
        [self.m_U, self.m_V] = self.orthonormal_basis_subspace_eigenvalue_1(young_tableau.projector())

    def __call__(self):
        return self.m_V.mult(self.m_data, operate_on="left") 

    def set(self, x):
        self.m_data = self.m_U.mult(x, operate_on="right") 



#############
### TESTS ###
#############

d = 4 # dimension

def test(tableau):
    test = np.empty(tuple([tableau.m_d for i in range(0, tableau.m_r)]))

    test = np.random.randint(256, size=tuple([tableau.m_d for i in range(0, tableau.m_r)]))
    test = np.tensordot(tableau.projector(),test,axes=tableau.m_r)
    print(test)

    tensor = Tensor(tableau)

    tensor.set(test)
    uncompressed_test = tensor()
    print(np.all(abs(uncompressed_test-test)<1e-10))

print("----- TESTS FOR MATRIXES -----")
# test sym 
tableau = YoungTableau([[1, 2]], d)
test(tableau)

# test antisym
tableau = YoungTableau([[1], [2]], d)
test(tableau)

print("----- TESTS FOR RANK-3 TENSORS -----")
# test sym 
tableau = YoungTableau([[1, 2, 3]], d)
test(tableau)

# test antisym
tableau = YoungTableau([[1],[2],[3]], d)
test(tableau)

# test mixed sym
tableau = YoungTableau([[1,2],[3]], d)
test(tableau)

tableau = YoungTableau([[1,3],[2]], d)
test(tableau)

print("----- TESTS FOR RANK-4 TENSORS -----")
# test Riemann tensor
tableau = YoungTableau([[1,2],[3,4]], d)
test(tableau)
