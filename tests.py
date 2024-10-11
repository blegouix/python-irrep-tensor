# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: GPL-3.0

import numpy as np

from young_tableau import *
from tensor import *

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
tableau = YoungTableau([[1,3],[2,4]], d)
test(tableau)
