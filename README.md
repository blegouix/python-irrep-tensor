# python-irrep-tensor
Represent any tensor exhibiting internal symmetries (belong to a finite group) with optimal memory footprint (store only independent components)

Relies on Young Tableau. Ie. :

```
tableau = YoungTableau([[1,2],[3]], 4) # Build Young tableau for mixed symmetric rank-3 tensors of dimension 4
projector = tableau.projector() # Compute the associated Young projector (dimension 8)

test = np.random.randint(256, size=(4, 4, 4)) # Generate a random dimension 4 rank-3 tensor
test = np.tensordot(projector, test, axes=3) # Project the tensor to extract its mixed-symmetric part (corresponding to the Young tableau defined above)

tensor = Tensor(tableau) # Build a Tensor object which corresponds to a representation of the finite group associated to the Young tableau
tensor.set(test) # Fill the Tensor object with the test tensor values
# AT THIS POINT, THE TENSOR IS REPRESENTED THROUGH INDEPENDENT COMPONENTS ONLY (check tensor.m_data)
uncompressed_test = tensor() # Recompute all the components of the test tensor
print(np.all(abs(uncompressed_test-test)<1e-10)) # True
```

Run ```python3 tests.py``` to check correct behaviour for rank-2 to rank-3 representations.

## References

- [Group Theory, Birdtracks, Lie’s, and Exceptional Groups](https://birdtracks.eu/), Predrag Cvitanović, Princeton University Press July 2008.
