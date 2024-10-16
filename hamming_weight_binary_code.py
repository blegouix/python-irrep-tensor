# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: GPL-3.0

import itertools

def index_hamming_weight_code(n, length):
    count = 0
    # Iterate through numbers by increasing Hamming weight
    for hamming_weight in range(0, length):
        # Generate all binary numbers with the given Hamming weight
        for bits in itertools.combinations(range(length), hamming_weight):
            num = 0
            for bit in bits:
                num |= (1 << bit) 
            if count == n:
                return num
            count = count + 1
    return -1
