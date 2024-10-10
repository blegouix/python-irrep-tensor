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
        for row in self.m_tableau:
            if len(row)>=2:
                Tr = np.zeros(tuple([self.m_d for j in range(0, 2*self.m_r)]))
                row_ = [elem-1 for elem in row]
                row_.reverse()
                fill(Tr, (), tr_lambda(row_))
                Sym = (Tr+I)/2
                proj = Sym
        tableau_dual = dual_syt(self.m_tableau)
        for row in tableau_dual:
            if len(row)>=2:
                Tr = np.zeros(tuple([self.m_d for j in range(0, 2*self.m_r)]))
                row_ = [elem-1 for elem in row]
                row_.reverse()
                fill(Tr, (), tr_lambda(row_))
                AntiSym = (Tr-I)/2
                proj = AntiSym
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

# Projector filler
def fill(T, idx, lambda_func):
    r = len(T.shape)
    d = T.shape[0]
    for i in range(0,d):
        new_idx = idx + (i,)
        if len(new_idx)==r:
            lambda_func(T, new_idx)
        else:
            fill(T, new_idx, lambda_func)


#############
### TESTS ###
#############

print("----- TESTS FOR MATRIXES -----")

d = 4 # dimension

# Create rank-2 identity operator
def I_lambda(T, idx):
    if idx[0]==idx[2] and idx[1]==idx[3]:
        T[idx] = 1
I = np.zeros((d, d, d, d))
fill(I, (), I_lambda)

# Create rank-2 transpose operator
def Tr_lambda(T, idx):
    if idx[1]==idx[2] and idx[0]==idx[3]:
        T[idx] = 1
Tr = np.zeros((d, d, d, d))
fill(Tr, (), Tr_lambda)

# Create symmetric projection
Sym = (Tr+I)/2

# Create antisymmetric projection
AntiSym = (Tr-I)/2 

#test sym
tensor = Tensor(YoungTableau([[1, 2]], d))

test = np.empty((d,d))

for i in range(0,d):
    for j in range(0,d):
        test[i,j] = i+j 
print(test)

tensor.set(test)
uncompressed_test = tensor()
print(np.all(abs(uncompressed_test-test)<1e-14))

#test antisym
tensor = Tensor(YoungTableau([[1],[2]], d))

test = np.empty((d,d))

for i in range(0,d):
    for j in range(0,d):
        test[i,j] = j-i
print(test)

tensor.set(test)
uncompressed_test = tensor()
print(np.all(abs(uncompressed_test-test)<1e-14))

print("----- TESTS FOR RANK-3 TENSORS -----")

d = 4 # dimension

# Create rank-3 identity operator
def I_lambda(T, idx):
    if idx[0]==idx[3] and idx[1]==idx[4] and idx[2]==idx[5]:
        T[idx] = 1
I = np.zeros((d, d, d, d, d, d))
fill(I, (), I_lambda)

# Create rank-3 transpose operators
def Tr12_lambda(T, idx):
    if idx[0]==idx[3] and idx[2]==idx[4] and idx[1]==idx[5]:
        T[idx] = 1
Tr12 = np.zeros((d, d, d, d, d, d))
fill(Tr12, (), Tr12_lambda)

def Tr01_lambda(T, idx):
    if idx[1]==idx[3] and idx[0]==idx[4] and idx[2]==idx[5]:
        T[idx] = 1
Tr01 = np.zeros((d, d, d, d, d, d))
fill(Tr01, (), Tr01_lambda)

def Tr120_lambda(T, idx):
    if idx[1]==idx[3] and idx[2]==idx[4] and idx[0]==idx[5]:
        T[idx] = 1
Tr120 = np.zeros((d, d, d, d, d, d))
fill(Tr120, (), Tr120_lambda)

def Tr201_lambda(T, idx):
    if idx[2]==idx[3] and idx[0]==idx[4] and idx[1]==idx[5]:
        T[idx] = 1
Tr201 = np.zeros((d, d, d, d, d, d))
fill(Tr201, (), Tr201_lambda)

def Tr02_lambda(T, idx):
    if idx[2]==idx[3] and idx[1]==idx[4] and idx[0]==idx[5]:
        T[idx] = 1
Tr02 = np.zeros((d, d, d, d, d, d))
fill(Tr02, (), Tr02_lambda)

# Create symmetric projection
Sym = (Tr12 + Tr01 + Tr02 + I + Tr120 + Tr201)/6 

# Create antisymmetric projection
AntiSym = (Tr12 + Tr01 + Tr02 - I - Tr120 - Tr201)/6 

# Create rank-3 specific projection (dont know the name of it)
Sym01 = (Tr01 + I)/2
AntiSym12 = (Tr12 - I)/2
# MixedSym = 4/3*Sym01*AntiSym12*Sym01
MixedSym = I+Tr01-Tr02-Tr120

# test sym 
tensor = Tensor(YoungTableau([[1, 2, 3]], d))

test = np.empty((d,d,d))

test = np.random.randint(256, size=(d,d,d))
test = np.tensordot(Sym,test,axes=3)
print(test)

tensor.set(test)
uncompressed_test = tensor()
print(np.all(abs(uncompressed_test-test)<1e-13))

# test antisym
tensor = Tensor(YoungTableau([[1],[2],[3]], d))

test = np.empty((d,d,d))

test = np.random.randint(256, size=(d,d,d))
test = np.tensordot(AntiSym,test,axes=3)
print(test)

tensor.set(test)
uncompressed_test = tensor()
print(np.all(abs(uncompressed_test-test)<1e-13))

# test mixed sym
print("-----")
tensor = Tensor(YoungTableau([[1,2],[3]], d))

test = np.empty((d,d,d))

test = np.random.randint(256, size=(d,d,d))
test = np.tensordot(MixedSym, test, axes=3)
print(test)

tensor.set(test)
uncompressed_test = tensor()
print(np.all(abs(uncompressed_test-test)<1e-12))
