import numpy as np
import itertools

# COO storage
class Coo:
    m_ndims = np.empty(0) 
    m_idx = np.empty(0) 
    m_values = np.empty(0)

    def __init__(self, ndims):
        self.m_ndims = np.array(ndims) 
        self.m_idx = np.empty((len(ndims), 0))
        self.m_values = np.empty(0)

    def copy_construct(self, ndims, idx, values):
        self.m_ndims = np.array(ndims) 
        self.m_idx = idx
        self.m_values = np.array(values) 
    
    def append_dense(self, dense, **kwargs):
        idx = np.nonzero(dense)
        self.m_values = np.append(self.m_values, dense[idx])
        idx = np.array(idx)
        for i in range(0, idx.shape[1]):
            idx[kwargs.get('axis', None), i] = self.m_ndims[kwargs.get('axis', None)]
        self.m_ndims[kwargs.get('axis', None)] = self.m_ndims[kwargs.get('axis', None)]+1
        self.m_idx = np.append(self.m_idx, idx, 1)

    def append_coo(self, coo, **kwargs):
        self.m_values = np.append(self.m_values, coo.m_values)
        self.m_ndims[kwargs.get('axis', None)] = self.m_ndims[kwargs.get('axis', None)]+1
        self.m_idx = np.append(self.m_idx, coo.m_idx, 1)

    # /!\ support only coalescent axis
    def get(self, id, **kwargs):
        i = 0
        ndims = np.copy(self.m_ndims)
        ndims[kwargs.get('axis', None)] = 0
        coo = Coo(ndims)
        coo_ = Coo(ndims) # trick to enforce correct indexing along axis
        while i<self.m_idx.shape[1] and self.m_idx[kwargs.get('axis', None), i]<=id:
            if self.m_idx[kwargs.get('axis', None), i]==id:
                coo_to_add = Coo(ndims)
                coo_to_add.copy_construct(ndims, np.expand_dims(self.m_idx[:,i], axis=-1), self.m_values[i])
                coo_.append_coo(coo_to_add, axis=kwargs.get('axis', None))
            i = i+1
        coo.append_coo(coo_, axis=kwargs.get('axis', None)) 
        
        #enforce id=0 in the axis orthogonal to the slice
        coo.m_idx[kwargs.get('axis', None), :] = 0
        return coo 

    # /!\ support only coalescent axis
    def __mul__(self, x):
        prod = np.zeros(self.m_ndims[:-1])
        for k in range(0, self.m_ndims[-1]):
            slice = self.get(k, axis=-1)
            x_k = x[k]
            for l in range(0, slice.m_idx.shape[1]):
                id = tuple(slice.m_idx[:-1,l].astype(int));
                prod[id] = prod[id] + x_k*slice.m_values[l]
        return prod

    # /!\ support only coalescent axis
    def __matmul__(self, x):
        prod = np.zeros(self.m_ndims[0])
        for k in range(0, self.m_ndims[0]):
            slice = self.get(k, axis=0)
            for l in range(0, slice.m_idx.shape[1]):
                id = tuple(slice.m_idx[1:,l].astype(int));
                prod[k] = prod[k] + slice.m_values[l]*x[id]
        return prod
                
def coo2dense(coo):
    dense = np.zeros(coo.m_ndims);
    for i in range(0, np.size(coo.m_values)):
        dense[tuple([int(coo.m_idx[j, i]) for j in range(0,len(coo.m_ndims))])] = coo.m_values[i]
    return dense

# General Tensor class with internal symmetries
class Tensor:
    m_r = 0
    m_d = 0
    m_U = np.empty(0) 
    m_V = np.empty(0) 
    m_data = np.empty(0) 

    # Gram-shmidt approach to produce from vec an orthogonal vector to the vector space generated by basis
    def orthogonalize(self, vec, basis):
        for i in range(0,basis.m_ndims[-1]):
            eigentensor = coo2dense(basis.get(i, axis=-1)).reshape(self.m_d,self.m_d)
            vec = vec - np.tensordot(vec, eigentensor)/np.tensordot(eigentensor, eigentensor)*eigentensor
        return vec

    # Build orthonormal basis for the eigen subspace associated to the eigenvalue 1 of the projection operator
    def orthonormal_basis_subspace_eigenvalue_1(self, Proj):
        U = Coo((0,self.m_d,self.m_d)); 
        V = Coo((self.m_d,self.m_d,0)); 

        index = 0;
        # while np.size(V, axis=2)<d*(d+1)//2:
        while V.m_ndims[2]<self.m_d*(self.m_d-1)//2: # TODO: general formula
            index = index+1;
            candidate = np.zeros((self.m_d, self.m_d))
            for i in range(0,self.m_d):
                for j in range(0,self.m_d):
                    candidate[i,j] = index//2**(i*self.m_d+j)%2
            # candidate = np.tensordot(Proj, np.random.randint(2, size=(d,d)));
            v = self.orthogonalize(np.tensordot(Proj, candidate), V);

            if np.any(v>0.25):
                v = v/np.sqrt(np.tensordot(v, v)) # normalize
                u = np.tensordot(np.tensordot(Tr, Proj), v) # not sure if this or u = v is correct
                # u = v 
                U.append_dense(u.reshape(1,self.m_d,self.m_d), axis=0)
                V.append_dense(v.reshape(self.m_d,self.m_d,1), axis=-1)
        return [U, V] 

    def __init__(self, Proj):
        # TODO: assert Proj hypercubic and pair rank
        self.m_r = len(Proj.shape)//2
        self.m_d = Proj.shape[0]
        [self.m_U, self.m_V] = self.orthonormal_basis_subspace_eigenvalue_1(Proj)

    def __call__(self):
        return self.m_V*self.m_data 

    def set(self, x):
        self.m_data = self.m_U@x 
        # self.m_data = np.tensordot(coo2dense(self.m_U), x, axes=self.m_r)

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

def I_lambda(T, idx):
    if idx[0]==idx[2] and idx[1]==idx[3]:
        T[idx] = 1

def Tr_lambda(T, idx):
    if idx[0]==idx[3] and idx[1]==idx[2]:
        T[idx] = 1


# main
d = 4 # dimension

# Create rank-2 identity operator
I = np.zeros((d, d, d, d))
fill(I, (), I_lambda)

# Create rank-2 transpose operator
Tr = np.zeros((d, d, d, d))
fill(Tr, (), Tr_lambda)

# Create symmetric projection
Sym = (Tr+I)/2

# Create antisymmetric projection
AntiSym = (Tr-I)/2 

#test
tensor = Tensor(AntiSym)

test = np.empty((d,d))

for i in range(0,d):
    for j in range(0,d):
        test[i,j] = j-i
print(test)

tensor.set(test)
uncompressed_test = tensor()
print(np.all(abs(uncompressed_test-test)<1e-14))
