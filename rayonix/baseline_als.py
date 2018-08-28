import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

#lmb and p parameters may vary for each case
def baseline_als(y, lmb=1e6, p=1e-6, niter=10):
	"""Based on algorithm developed and reported by P.H.C. 'Eilers and H.F.M. Boelens - Baseline Correction with Asymmetric Least Squares Smoothing, 2005.
	Implemented code was found on http://stackoverflow.com/questions/29156532/python-baseline-correction-library"""
	
	L = len(y)
	D = sparse.csc_matrix(np.diff(np.eye(L), 2))
	w = np.ones(L)
	for i in range(niter):
		W = sparse.spdiags(w, 0, L, L)
		Z = W + lmb * D.dot(D.transpose())
		bck = spsolve(Z, w*y)
		w = p * (y > bck) + (1-p) * (y < bck)
	
	return bck