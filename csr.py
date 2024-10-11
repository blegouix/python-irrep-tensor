import numpy as np

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
