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
