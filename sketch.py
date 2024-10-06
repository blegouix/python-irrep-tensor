import numpy as np

d = 4; # dimension

# Create rank-2 identity operator
I = np.zeros((d, d, d, d));

for i in range(0,d):
    for j in range(0,d):
        for k in range(0,d):
            for l in range(0,d):
                if i==k and j==l:
                    I[i,j,k,l] = 1

# Create rank-2 transpose operator
Tr = np.zeros((d, d, d, d));

for i in range(0,d):
    for j in range(0,d):
        for k in range(0,d):
            for l in range(0,d):
                if i==l and j==k:
                    Tr[i,j,k,l] = 1
# Create symmetric projection
Sym = (Tr+I)/2

# Create symmetric projection
AntiSym = (Tr-I)/2

# Gram-shmidt-like approach to determine if a vector is out of a vector space
def is_out_of_vector_space(vec, basis):
    for i in range(0,basis.shape[-1]):
        vec = vec - np.tensordot(vec, basis[:,:,i].reshape(d,d))/np.tensordot(basis[:,:,i].reshape(d,d), basis[:,:,i].reshape(d,d))*basis[:,:,i].reshape(d,d)
    return np.any(vec != 0);

# Build orthonormal basis for the eigen subspace associated to the eigenvalue 1 of the projection operator
def orthonormal_basis_subspace_eigenvalue_1(Proj):
    basis = np.empty((d,d,0)); 
    for index in range(1,2**(d**2)):
        candidate = np.zeros((d, d))
        for i in range(0,d):
            for j in range(0,d):
                candidate[i,j] = index//2**(i*d+j)%2
        if (np.all(np.tensordot(Proj-I, candidate, axes=2) == 0) and is_out_of_vector_space(candidate,basis)):
            basis = np.append(basis, candidate.reshape(d,d,1), axis=2)
    return basis

#test
test = np.zeros(d*(d+1)//2);

for i in range(0,d*(d+1)//2):
        test[i] = i

print(np.tensordot(orthonormal_basis_subspace_eigenvalue_1(Sym), test, axes=1))
