#include "detectum.h"

// Householder transformation X(i0:iend,j0:jend) = H*X(i0:iend,j0:jend),
// where H = I - beta*v*v' and v(0) = 1. stride is the increment of v.
static void housef_apply_l(Matrixf* X, const float* v, float beta,
	int i0, int iend, int j0, int jend, int stride)
{
	int i, j;
	float h, * colXj;

	if (beta != 0) {
		for (j = j0; j <= jend; j++) {
			colXj = &at(X, 0, j);
			h = colXj[i0];
			for (i = i0 + 1; i <= iend; i++) {
				h += colXj[i] * v[(i - i0) * stride];
			}
			h *= beta;
			colXj[i0] -= h;
			for (i = i0 + 1; i <= iend; i++) {
				colXj[i] -= h * v[(i - i0) * stride];
			}
		}
	}
}

// Householder transformation X(i0:iend,j0:jend) = X(i0:iend,j0:jend)*H,
// where H = I - beta*v*v' and v(0) = 1. stride is the increment of v.
static void housef_apply_r(Matrixf* X, const float* v, float beta,
	int i0, int iend, int j0, int jend, int stride)
{
	int i, j;
	float h;

	if (beta != 0) {
		for (i = i0; i <= iend; i++) {
			h = at(X, i, j0);
			for (j = j0 + 1; j <= jend; j++) {
				h += at(X, i, j) * v[(j - j0) * stride];
			}
			h *= beta;
			at(X, i, j0) -= h;
			for (j = j0 + 1; j <= jend; j++) {
				at(X, i, j) -= h * v[(j - j0) * stride];
			}
		}
	}
}

void matrixf_init(Matrixf* A, int rows, int cols, float* data, int ordmem)
{
	A->data = data;
	if (ordmem) {
		A->rows = cols;
		A->cols = rows;
		matrixf_transpose(A);
	}
	else {
		A->rows = rows;
		A->cols = cols;
	}
}

int matrixf_permute(Matrixf* A, Matrixf* p, int reverseP, int transP)
{
	int i, j, k, q;
	const int len = p->rows * p->cols;
	float t, * x = p->data, * colAi, * colAj;

	if (reverseP) {
		q = p->rows;
		p->rows = p->cols;
		p->cols = q;
	}
	if ((A->rows != p->rows || p->cols != 1) &&
		(A->cols != p->cols || p->rows != 1)) {
		return -1;
	}
	if ((reverseP && !transP) ||
		(!reverseP && transP)) {
		if (len > 1) {
			for (k = 0; k < len; k++) {
				i = (int)x[k];
				if (i >= 0) {
					j = k;
					do {
						q = (int)x[i];
						x[i] = (float)(-j - 1);
						j = i;
						i = q;
					} while (j != k);
				}
			}
			for (k = 0; k < len; k++) {
				x[k] = -x[k] - 1;
			}
		}
	}
	if (p->cols == 1) {
		matrixf_transpose(A);
	}
	q = A->rows;
	for (i = 0; i < len - 1; i++) {
		j = (int)x[i];
		while (j < i) {
			j = (int)x[j];
		}
		if (i != j) {
			colAi = &at(A, 0, i);
			colAj = &at(A, 0, j);
			for (k = 0; k < q; k++) {
				t = colAi[k];
				colAi[k] = colAj[k];
				colAj[k] = t;
			}
		}
	}
	if (p->cols == 1) {
		matrixf_transpose(A);
	}
	return 0;
}

void matrixf_transpose(Matrixf* A)
{
	int i, j;
	float t, * d = A->data;
	const int r = A->rows;
	const int c = A->cols;
	const int n = r * c - 1;

	for (j = i = 1; i < n; j = ++i) {
		do {
			j = j * r - n * (j / c);
		} while (j < i);
		t = d[i];
		d[i] = d[j];
		d[j] = t;
	}
	A->rows = c;
	A->cols = r;
}

int matrixf_decomp_chol(Matrixf* A)
{
	int i, j, k;
	const int n = A->rows;
	float v, r = 0;
	float* colAi, * colAj;

	if (A->cols != n) {
		return -1;
	}
	for (j = 0; j < n; j++) {
		for (i = j; i < n; i++) {
			colAi = &at(A, 0, i);
			colAj = &at(A, 0, j);
			v = colAi[j];
			for (k = 0; k < j; k++) {
				v -= colAi[k] * colAj[k];
			}
			if (i == j) {
				if (v > 0) {
					r = sqrtf(v);
				}
				else {
					return -2;
				}
			}
			colAi[j] = v / r;
		}
	}
	return 0;
}

int matrixf_decomp_lu(Matrixf* A, Matrixf* B)
{
	int i, j, k;
	const int n = A->rows;
	const int p = B->cols;
	float a, b, t;
	float* colXk, * colXi;

	if (A->cols != n || B->rows != n) {
		return -1;
	}
	matrixf_transpose(A);
	matrixf_transpose(B);
	for (i = 0; i < n - 1; i++) {
		k = j = i;
		a = fabsf(at(A, i, j));
		for (j = i + 1; j < n; j++) {
			b = fabsf(at(A, i, j));
			if (b > a) {
				k = j;
				a = b;
			}
		}
		colXk = &at(A, 0, k);
		colXi = &at(A, 0, i);
		for (j = 0; j < n; j++) {
			t = colXk[j];
			colXk[j] = colXi[j];
			colXi[j] = t;
		}
		colXk = &at(B, 0, k);
		colXi = &at(B, 0, i);
		for (j = 0; j < p; j++) {
			t = colXk[j];
			colXk[j] = colXi[j];
			colXi[j] = t;
		}
		a = at(A, i, i);
		for (k = i + 1; k < n; k++) {
			if (a != 0) {
				at(A, i, k) /= a;
			}
			b = at(A, i, k);
			colXk = &at(A, 0, k);
			colXi = &at(A, 0, i);
			for (j = i + 1; j < n; j++) {
				colXk[j] -= colXi[j] * b;
			}
		}
	}
	matrixf_transpose(A);
	matrixf_transpose(B);
	return 0;
}

int matrixf_decomp_lu_banded(Matrixf* A, int ubw)
{
	int i, j, p, piv;
	const int n = A->rows;
	float tau, t;

	if (A->cols != n || ubw < 0) {
		return -1;
	}
	for (i = 0; i < n - 1; i++) {
		piv = 0;
		p = (i + ubw + 2 < n) ? i + ubw + 2 : n;
		if (fabsf(at(A, i, i)) < fabsf(at(A, i + 1, i))) {
			for (j = i; j < p; j++) {
				t = at(A, i, j);
				at(A, i, j) = at(A, i + 1, j);
				at(A, i + 1, j) = t;
			}
			piv = 10;
		}
		if (at(A, i, i) != 0) {
			tau = at(A, i + 1, i) / at(A, i, i);
			for (j = i + 1; j < p; j++) {
				at(A, i + 1, j) -= tau * at(A, i, j);
			}
			at(A, i + 1, i) = tau + piv;
		}
	}
	return 0;
}

int matrixf_unpack_lu_banded(Matrixf* A, Matrixf* B)
{
	int i, j, piv;
	const int n = A->rows;
	const int p = B->cols;
	float t, tau;

	if (B->rows != n) {
		return -1;
	}
	for (i = 0; i < n - 1; i++) {
		piv = fabsf(at(A, i + 1, i)) > 1;
		tau = at(A, i + 1, i) - 10 * piv;
		for (j = 0; j < p; j++) {
			if (piv > 0) {
				t = at(B, i, j);
				at(B, i, j) = at(B, i + 1, j);
				at(B, i + 1, j) = t;
			}
			at(B, i + 1, j) -= tau * at(B, i, j);
		}
	}
	return 0;
}

int matrixf_decomp_qr(Matrixf* A, Matrixf* Q, Matrixf* P, Matrixf* B)
{
	int i, jm = 0, j = 0, k = 0, q = 0;
	const int m = A->rows;
	const int n = A->cols;
	const int kmax = m < n ? m - 1 : n - 1;
	float beta, t, c, cm, * v;
	float* colAk, * colAj, * colAjm;
	Matrixf A_econ = { n, n, A->data };

	if (Q) {
		if (Q->rows != m || (Q->cols != m && Q->cols != n) ||
			(Q->cols == n && m < n)) {
			return -1;
		}
	}
	if (P) {
		if ((P->rows != 1 && P->rows != n) || P->cols != n) {
			return -1;
		}
		for (j = 0; j < n; j++) {
			P->data[j] = (float)j;
		}
	}
	if (B && B->rows != m) {
		return -1;
	}
	for (k = 0; k <= kmax; k++) {
		colAk = &at(A, 0, k);
		if (P) {
			for (cm = 0, jm = 0, j = k; j < n; j++) {
				colAj = &at(A, 0, j);
				for (c = 0, i = k; i < m; i++) {
					c += colAj[i] * colAj[i];
				}
				if (c > cm) {
					cm = c;
					jm = j;
				}
			}
			if (cm == 0) {
				break;
			}
			colAjm = &at(A, 0, jm);
			for (j = 0; j < m; j++) {
				t = colAk[j];
				colAk[j] = colAjm[j];
				colAjm[j] = t;
			}
			t = P->data[k];
			P->data[k] = P->data[jm];
			P->data[jm] = t;
		}
		v = colAk + k;
		beta = housef(v, m - k, 1);
		housef_apply_l(A, v, beta, k, m - 1, k + 1, n - 1, 1);
		if (B) {
			housef_apply_l(B, v, beta, k, m - 1, 0, B->cols - 1, 1);
		}
	}
	if (Q) {
		q = Q->cols;
		for (i = 0; i < m * q; i++) {
			Q->data[i] = !(i % (m + 1));
		}
		matrixf_unpack_house(A, Q, 0, 0);
		if (q < m) {
			for (j = 0; j < n; j++) {
				for (i = 0; i < n; i++) {
					at(&A_econ, i, j) = (i <= j) ? at(A, i, j) : 0;
				}
			}
			A->rows = n;
		}
		else {
			for (j = 0; j < n; j++) {
				for (i = j + 1; i < m; i++) {
					at(A, i, j) = 0;
				}
			}
		}
	}
	if (P && P->rows == n) {
		for (i = n; i < n * n; i++) {
			P->data[i] = 0;
		}
		for (i = n - 1; i >= 0; i--) {
			j = (int)P->data[i];
			at(P, i, 0) = 0;
			at(P, j, i) = 1;
		}
	}
	return 0;
}

int matrixf_unpack_house(Matrixf* A, Matrixf* B, int s, int fwd)
{
	int i, k;
	const int f = fwd > 0 ? 1 : -1;
	const int m = A->rows;
	const int n = A->cols;
	const int p = B->cols;
	const int kmax = m - 1 < n ? m - 2 - s : n - 1 - s;
	float gamma, * v;

	if (B->rows != m || s < 0) {
		return -1;
	}
	for (k = f > 0 ? 0 : kmax; f > 0 ? k <= kmax : k >= 0; k += f) {
		v = &at(A, 0, k);
		for (gamma = 1.0f, i = k + 1 + s; i < m; i++) {
			gamma += v[i] * v[i];
		}
		if (gamma > 1) {
			housef_apply_l(B, v + k + s, 2.0f / gamma, k + s, m - 1, 0, p - 1, 1);
		}
	}
	return 0;
}

int matrixf_decomp_bidiag(Matrixf* A, Matrixf* U, Matrixf* V)
{
	int i, j, k, q = A->rows;
	const int m = A->rows;
	const int n = A->cols;
	const int kmax = m - 1 < n ? m - 2 : n - 1;
	float beta, * v;
	Matrixf A_econ = { n, n, A->data };

	if (m < n) {
		matrixf_transpose(A);
		k = matrixf_decomp_bidiag(A, V, U);
		matrixf_transpose(A);
		return k;
	}
	if ((U && (U->rows != m || (U->cols != m && U->cols != n))) ||
		(V && (V->rows != n || (V->cols != n)))) {
		return -1;
	}
	for (k = 0; k <= kmax; k++) {
		v = &at(A, k, k);
		beta = housef(v, m - k, 1);
		housef_apply_l(A, v, beta, k, m - 1, k + 1, n - 1, 1);
		if (k + 2 < n) {
			v = &at(A, k, k + 1);
			beta = housef(v, n - k - 1, m);
			housef_apply_r(A, v, beta, k + 1, m - 1, k + 1, n - 1, m);
		}
	}
	if (U) {
		q = U->cols;
		for (i = 0; i < m * q; i++) {
			U->data[i] = !(i % (m + 1));
		}
		matrixf_unpack_house(A, U, 0, 0);
		if (q < m) {
			for (j = 0; j < n; j++) {
				for (i = 0; i < n; i++) {
					at(&A_econ, i, j) = (i <= j) ? at(A, i, j) : 0;
				}
			}
			A->rows = n;
		}
		else {
			for (j = 0; j < n; j++) {
				for (i = j + 1; i < m; i++) {
					at(A, i, j) = 0;
				}
			}
		}
	}
	if (V) {
		matrixf_transpose(A);
		for (i = 0; i < n * n; i++) {
			V->data[i] = !(i % (n + 1));
		}
		matrixf_unpack_house(A, V, 1, 0);
		for (j = 0; j < A->cols; j++) {
			for (i = j + 2; i < A->rows; i++) {
				at(A, i, j) = 0;
			}
		}
		matrixf_transpose(A);
	}
	return 0;
}

int matrixf_decomp_cod(Matrixf* A, Matrixf* U, Matrixf* V, Matrixf* P, float tol)
{
	int i, j, rank = 0;
	const int m = A->rows;
	const int n = A->cols;
	const int p = m < n ? m : n;
	const int q = m > n ? m : n;

	if (matrixf_decomp_qr(A, U, P, 0)) {
		return -1;
	}
	if (!U) {
		for (j = 0; j < n; j++) {
			for (i = j + 1; i < m; i++) {
				at(A, i, j) = 0;
			}
		}
	}
	if (tol < 0) {
		tol = q * epsf(at(A, 0, 0));
	}
	while (rank < p && fabsf(at(A, rank, rank)) > tol) {
		rank++;
	}
	matrixf_transpose(A);
	A->cols = rank;
	if (matrixf_decomp_qr(A, V, 0, 0)) {
		return -1;
	}
	A->cols = U ? U->cols : m;
	matrixf_transpose(A);
	return rank;
}

#ifndef DETECTUM_SVD_ITER_MAX
#define DETECTUM_SVD_ITER_MAX (100 * n)
#endif
#ifndef DETECTUM_SVD_TOL
#define DETECTUM_SVD_TOL (3e-7f)
#endif
int matrixf_decomp_svd(Matrixf* A, Matrixf* U, Matrixf* V)
{
	const int m = A->rows;
	const int n = A->cols;
	int i, j, k, q, r = n - 1, iter = 0;
	const int iter_max = DETECTUM_SVD_ITER_MAX;
	const float tol = DETECTUM_SVD_TOL;
	float small, norm1, tmp, cosine, sine, Xi, Xj;
	float c00, c01, c11, y, z, mu;
	float* s = A->data, * p = 0;
	float* colXi, * colXj;
	Matrixf perm = { 1, n, p };

	if (m < n) {
		matrixf_transpose(A);
		k = matrixf_decomp_svd(A, V, U);
		matrixf_transpose(A);
		return k;
	}
	if (matrixf_decomp_bidiag(A, U, V)) {
		return -1;
	}
	q = A->rows;
	for (j = 0; j < n; j++) {
		for (i = 0; i < q; i++) {
			if (i != j && (i + 1) != j) {
				at(A, i, j) = 0;
			}
		}
	}
	for (norm1 = fabsf(at(A, 0, 0)), j = 1; j < n; j++) {
		tmp = fabsf(at(A, j - 1, j)) + fabsf(at(A, j, j));
		if (norm1 < tmp) {
			norm1 = tmp;
		}
	}
	small = tol * norm1;
	while (r > 0 && iter < iter_max) {
		while (r > 0 && fabsf(at(A, r - 1, r)) <= small) {
			r--;
		}
		if (r > 0) {
			q = i = r;
			while (q > 0 && fabsf(at(A, q - 1, q)) > small) {
				q--;
			}
			while (i >= q && fabsf(at(A, i, i)) > small) {
				i--;
			}
			if (i == q - 1) {
				c00 = c01 = c11 = 0;
				colXi = &at(A, 0, r - 1);
				colXj = &at(A, 0, r);
				for (k = (r - 2 < 0 ? 0 : r - 2); k <= r; k++) {
					Xi = colXi[k];
					Xj = colXj[k];
					c00 += Xi * Xi;
					c01 += Xi * Xj;
					c11 += Xj * Xj;
				}
				y = 0.5f * (c00 - c11);
				z = sqrtf(y * y + c01 * c01);
				mu = fabsf(y + z) < fabsf(y - z) ? c11 + y + z : c11 + y - z;
				y = at(A, q, q) * at(A, q, q) - mu;
				z = at(A, q, q + 1) * at(A, q, q);
				for (j = q; j < r; j++) {
					givensf(y, z, &cosine, &sine);
					colXi = &at(A, 0, j);
					colXj = &at(A, 0, j + 1);
					for (k = (j - 1 < 0 ? 0 : j - 1); k <= j + 1; k++) {
						Xi = colXi[k];
						Xj = colXj[k];
						colXi[k] = cosine * Xi - sine * Xj;
						colXj[k] = sine * Xi + cosine * Xj;
					}
					if (V) {
						colXi = &at(V, 0, j);
						colXj = &at(V, 0, j + 1);
						for (k = 0; k < n; k++) {
							Xi = colXi[k];
							Xj = colXj[k];
							colXi[k] = cosine * Xi - sine * Xj;
							colXj[k] = sine * Xi + cosine * Xj;
						}
					}
					y = at(A, j, j);
					z = at(A, j + 1, j);
					givensf(y, z, &cosine, &sine);
					for (k = j; k <= (j + 2 < r ? j + 2 : r); k++) {
						Xi = at(A, j, k);
						Xj = at(A, j + 1, k);
						at(A, j, k) = cosine * Xi - sine * Xj;
						at(A, j + 1, k) = sine * Xi + cosine * Xj;
					}
					if (U) {
						colXi = &at(U, 0, j);
						colXj = &at(U, 0, j + 1);
						for (k = 0; k < m; k++) {
							Xi = colXi[k];
							Xj = colXj[k];
							colXi[k] = cosine * Xi - sine * Xj;
							colXj[k] = sine * Xi + cosine * Xj;
						}
					}
					if (j < r - 1) {
						y = at(A, j, j + 1);
						z = at(A, j, j + 2);
					}
				}
				iter++;
			}
			else if (i < r) {
				for (j = i + 1; j <= r; j++) {
					givensf(-at(A, j, j), at(A, i, j), &cosine, &sine);
					for (k = j; k <= (j + 1 < r ? j + 1 : r); k++) {
						Xi = at(A, i, k);
						Xj = at(A, j, k);
						at(A, i, k) = cosine * Xi - sine * Xj;
						at(A, j, k) = sine * Xi + cosine * Xj;
					}
					if (U) {
						colXi = &at(U, 0, i);
						colXj = &at(U, 0, j);
						for (k = 0; k < m; k++) {
							Xi = colXi[k];
							Xj = colXj[k];
							colXi[k] = cosine * Xi - sine * Xj;
							colXj[k] = sine * Xi + cosine * Xj;
						}
					}
				}
			}
			else {
				for (j = r - 1; j >= q; j--) {
					givensf(at(A, j, j), at(A, j, r), &cosine, &sine);
					colXi = &at(A, 0, j);
					colXj = &at(A, 0, r);
					for (k = (j - 1 > q ? j - 1 : q); k <= j; k++) {
						Xi = colXi[k];
						Xj = colXj[k];
						colXi[k] = cosine * Xi - sine * Xj;
						colXj[k] = sine * Xi + cosine * Xj;
					}
					if (V) {
						colXi = &at(V, 0, j);
						colXj = &at(V, 0, r);
						for (k = 0; k < n; k++) {
							Xi = colXi[k];
							Xj = colXj[k];
							colXi[k] = cosine * Xi - sine * Xj;
							colXj[k] = sine * Xi + cosine * Xj;
						}
					}
				}
			}
		}
	}
	if (n > 1) {
		p = A->data + A->rows;
		perm.data = p;
	}
	for (j = 0; j < n; j++) {
		s[j] = at(A, j, j);
		if (j > 1) {
			for (i = 0; i < n; i++) {
				at(A, i, j) = 0;
			}
		}
		if (s[j] < 0) {
			s[j] *= -1.0f;
			if (U) {
				for (i = 0; i < m; i++) {
					at(U, i, j) *= -1.0f;
				}
			}
		}
		if (p) {
			p[j] = (float)j;
		}
	}
	if (p) {
		for (i = 1; i < n; i++) {
			for (j = i; j > 0; j--) {
				if (s[j - 1] < s[j]) {
					tmp = s[j - 1];
					s[j - 1] = s[j];
					s[j] = tmp;
					tmp = p[j - 1];
					p[j - 1] = p[j];
					p[j] = tmp;
				}
			}
		}
		if (U) {
			k = U->cols;
			U->cols = n;
			matrixf_permute(U, &perm, 0, 0);
			U->cols = k;
		}
		if (V) {
			matrixf_permute(V, &perm, 0, 0);
		}
		p[0] = 0;
		for (j = 1; j < n; j++) {
			p[j] = 0;
			at(A, j, j) = s[j];
			s[j] = 0;
		}
	}
	return iter == iter_max ? -2 : iter;
}

#ifndef DETECTUM_SVD_JACOBI_ITER_MAX
#define DETECTUM_SVD_JACOBI_ITER_MAX (10 * n)
#endif
#ifndef DETECTUM_SVD_JACOBI_TOL
#define DETECTUM_SVD_JACOBI_TOL (3e-7f)
#endif
int matrixf_decomp_svd_jacobi(Matrixf* A, Matrixf* U, Matrixf* V)
{
	int i, j, k, count = 1, iter = 0, sorted, orthog;
	float a, b, p, q, v, Xij, Xik, s, sine, cosine;
	float* colXj, * colXk;
	const int m = A->rows;
	const int n = A->cols;
	const int iter_max = DETECTUM_SVD_JACOBI_ITER_MAX;
	const float tol = DETECTUM_SVD_JACOBI_TOL;

	if (m < n) {
		matrixf_transpose(A);
		k = matrixf_decomp_svd_jacobi(A, V, U);
		matrixf_transpose(A);
		return k;
	}
	if (U) {
		if (U->rows != m || (U->cols != m && U->cols != n)) {
			return -1;
		}
	}
	if (V) {
		if (V->rows != n || V->cols != n) {
			return -1;
		}
		for (i = 0; i < n * n; i++) {
			V->data[i] = !(i % (n + 1));
		}
	}
	while (count > 0 && iter < iter_max) {
		count = n * (n - 1) / 2;
		for (j = 0; j < n - 1; j++) {
			for (k = j + 1; k < n; k++) {
				a = b = p = 0;
				colXj = &at(A, 0, j);
				colXk = &at(A, 0, k);
				for (i = 0; i < m; i++) {
					Xij = colXj[i];
					Xik = colXk[i];
					a += Xij * Xij;
					b += Xik * Xik;
					p += Xij * Xik;
				}
				p *= 2.0f;
				q = a - b;
				v = hypotf(p, q);
				sorted = a >= b;
				orthog = fabsf(p) <= tol * sqrtf(a * b);
				if (sorted && orthog) {
					count--;
				}
				else {
					if (q < 0) {
						s = (p < 0) ? -1.0f : 1.0f;
						sine = sqrtf(0.5f * (1 - q / v)) * s;
						cosine = 0.5f * p / (v * sine);
					}
					else {
						cosine = sqrtf(0.5f * (1 + q / v));
						sine = 0.5f * p / (v * cosine);
					}
					for (i = 0; i < m; i++) {
						Xij = colXj[i];
						Xik = colXk[i];
						colXj[i] = +Xij * cosine + Xik * sine;
						colXk[i] = -Xij * sine + Xik * cosine;
					}
					if (V) {
						colXj = &at(V, 0, j);
						colXk = &at(V, 0, k);
						for (i = 0; i < n; i++) {
							Xij = colXj[i];
							Xik = colXk[i];
							colXj[i] = +Xij * cosine + Xik * sine;
							colXk[i] = -Xij * sine + Xik * cosine;
						}
					}
				}
			}
		}
		iter++;
	}
	if (U) {
		matrixf_decomp_qr(A, U, 0, 0);
		for (j = 0; j < n; j++) {
			if (at(A, j, j) < 0) {
				at(A, j, j) *= -1.0f;
				for (i = 0; i < m; i++) {
					at(U, i, j) *= -1.0f;
				}
			}
			for (i = 0; i < j; i++) {
				at(A, i, j) = 0;
			}
		}
	}
	return iter == iter_max ? -2 : iter;
}

int matrixf_decomp_hess(Matrixf* A, Matrixf* P)
{
	int i, j, k;
	const int n = A->rows;
	float beta, * v;

	if (A->cols != n || (P && (P->rows != n || P->cols != n))) {
		return -1;
	}
	for (k = 0; k < n - 2; k++) {
		v = &at(A, k + 1, k);
		beta = housef(v, n - k - 1, 1);
		housef_apply_l(A, v, beta, k + 1, n - 1, k + 1, n - 1, 1);
		housef_apply_r(A, v, beta, 0, n - 1, k + 1, n - 1, 1);
	}
	if (P) {
		for (i = 0; i < n * n; i++) {
			P->data[i] = !(i % (n + 1));
		}
		matrixf_unpack_house(A, P, 1, 0);
		for (j = 0; j < n; j++) {
			for (i = j + 2; i < n; i++) {
				at(A, i, j) = 0;
			}
		}
	}
	return 0;
}

#ifndef DETECTUM_SCHUR_SYMM_ITER_MAX
#define DETECTUM_SCHUR_SYMM_ITER_MAX (100 * n)
#endif
#ifndef DETECTUM_SCHUR_SYMM_TOL
#define DETECTUM_SCHUR_SYMM_TOL (3e-7f)
#endif
int matrixf_decomp_schur_symm(Matrixf* A, Matrixf* U)
{
	const int n = A->rows;
	int k, i, imin, imax, q, m = n - 1, iter = 0;
	const int iter_max = DETECTUM_SCHUR_SYMM_ITER_MAX;
	const float tol = DETECTUM_SCHUR_SYMM_TOL;
	float d, f, g, x, y, cosine, sine, Xk, Xk1;
	float* colXk, * colXk1;

	if (matrixf_decomp_hess(A, U)) {
		return -1;
	}
	if (!U) {
		for (k = 0; k < n - 2; k++) {
			at(A, k + 2, k) = 0;
		}
	}
	while (m > 0 && iter < iter_max) {
		while (m > 0 && at(A, m, m - 1) == 0) {
			m--;
		}
		if (m > 0) {
			q = m - 1;
			while (q > 0 && at(A, q, q - 1) != 0) {
				q--;
			}
			d = 0.5f * (at(A, m - 1, m - 1) - at(A, m, m));
			f = d < 0 ? -1.0f : 1.0f;
			g = at(A, m, m - 1) * at(A, m, m - 1);
			x = at(A, q, q) - at(A, m, m) + g / (d + f * sqrtf(d * d + g));
			y = at(A, q + 1, q);
			for (k = q; k < m; k++) {
				givensf(x, y, &cosine, &sine);
				imin = k - 1 < 0 ? 0 : k - 1;
				imax = k + 2 > n - 1 ? n - 1 : k + 2;
				for (i = imin; i <= imax; i++) {
					Xk = at(A, k, i);
					Xk1 = at(A, k + 1, i);
					at(A, k, i) = cosine * Xk - sine * Xk1;
					at(A, k + 1, i) = sine * Xk + cosine * Xk1;
				}
				for (i = imin; i <= imax; i++) {
					Xk = at(A, i, k);
					Xk1 = at(A, i, k + 1);
					at(A, i, k) = cosine * Xk - sine * Xk1;
					at(A, i, k + 1) = sine * Xk + cosine * Xk1;
				}
				if (U) {
					colXk = &at(U, 0, k);
					colXk1 = &at(U, 0, k + 1);
					for (i = 0; i < n; i++) {
						Xk = colXk[i];
						Xk1 = colXk1[i];
						colXk[i] = cosine * Xk - sine * Xk1;
						colXk1[i] = sine * Xk + cosine * Xk1;
					}
				}
				if (k < m - 1) {
					x = at(A, k + 1, k);
					y = at(A, k + 2, k);
				}
			}
			for (k = q; k < m; k++) {
				if (fabsf(at(A, k + 1, k)) <=
					tol * (fabsf(at(A, k, k)) + fabsf(at(A, k + 1, k + 1)))) {
					at(A, k + 1, k) = 0;
				}
			}
			iter++;
		}
	}
	for (k = 1; k < n * n - 1; k++) {
		if (k % (n + 1)) {
			A->data[k] = 0;
		}
	}
	return iter == iter_max ? -2 : iter;
}

#ifndef DETECTUM_SCHUR_ITER_MAX
#define DETECTUM_SCHUR_ITER_MAX (100 * n)
#endif
#ifndef DETECTUM_SCHUR_TOL
#define DETECTUM_SCHUR_TOL (3e-7f)
#endif
#ifndef DETECTUM_SCHUR_AD_HOC_SHIFT_COUNT
#define DETECTUM_SCHUR_AD_HOC_SHIFT_COUNT (5)
#endif
int matrixf_decomp_schur(Matrixf* A, Matrixf* U)
{
	const int n = A->rows;
	int i, j, k, q, m = n - 1, iter = 0, ad_hoc_shift;
	const int iter_max = DETECTUM_SCHUR_ITER_MAX;
	const int ahsc = DETECTUM_SCHUR_AD_HOC_SHIFT_COUNT;
	const float tol = DETECTUM_SCHUR_TOL;
	const float eps = epsf(1);
	float r, s, t, x, y, z, mu, beta, v[3] = { 1, 0, 0 };
	float sine, cosine, Xk, Xk1;
	float* colXk, * colXk1;

	if (matrixf_decomp_hess(A, U)) {
		return -1;
	}
	if (!U) {
		for (k = 0; k < n - 2; k++) {
			for (i = k + 2; i < n; i++) {
				at(A, i, k) = 0;
			}
		}
	}
	else {
		matrixf_transpose(U);
	}
	while (m > 1 && iter < iter_max) {
		while (m > 1 && (at(A, m, m - 1) == 0 || at(A, m - 1, m - 2) == 0)) {
			if (at(A, m, m - 1) == 0) {
				m--;
			}
			else if (at(A, m - 1, m - 2) == 0) {
				m -= 2;
			}
		}
		if (m > 0) {
			q = m - 1;
			while (q > 0 && at(A, q, q - 1) != 0) {
				q--;
			}
			ad_hoc_shift = !((iter + 1) % ahsc);
			if (m - q > 1) {
				if (ad_hoc_shift) {
					x = 0.5f * (at(A, m - 1, m - 1) - at(A, m, m));
					y = x * x + at(A, m - 1, m) * at(A, m, m - 1);
					s = t = x = at(A, m, m) + x;
					if (y > 0) {
						z = sqrtf(y);
						s -= z;
						t += z;
						x = fabsf(s - at(A, m, m)) < fabsf(t - at(A, m, m)) ? s : t;
					}
					if (x == 0) {
						x = eps;
					}
					s = x + x;
					t = x * x;
				}
				else {
					s = at(A, m - 1, m - 1) + at(A, m, m);
					t = at(A, m - 1, m - 1) * at(A, m, m) - at(A, m - 1, m) * at(A, m, m - 1);
				}
				x = at(A, q, q) * at(A, q, q) + at(A, q, q + 1) * at(A, q + 1, q) - s * at(A, q, q) + t;
				y = at(A, q + 1, q) * (at(A, q, q) + at(A, q + 1, q + 1) - s);
				z = at(A, q + 1, q) * at(A, q + 2, q + 1);
				for (k = q - 1; k <= m - 3; k++) {
					t = hypotf(x, y);
					mu = hypotf(t, z);
					if (mu > 0) {
						s = x >= 0 ? 1.0f : -1.0f;
						t = x + s * mu;
						v[1] = y / t;
						v[2] = z / t;
						beta = t * s / mu;
						housef_apply_l(A, v, beta, k + 1, k + 3, q > k ? q : k, n - 1, 1);
						housef_apply_r(A, v, beta, 0, (k + 4) < m ? (k + 4) : m, k + 1, k + 3, 1);
						if (U) {
							housef_apply_l(U, v, beta, k + 1, k + 3, 0, n - 1, 1);
						}
					}
					x = at(A, k + 2, k + 1);
					y = at(A, k + 3, k + 1);
					if (k < m - 3) {
						z = at(A, k + 4, k + 1);
					}
				}
				mu = hypotf(x, y);
				if (mu > 0) {
					s = x >= 0 ? 1.0f : -1.0f;
					t = x + s * mu;
					v[1] = y / t;
					beta = t * s / mu;
					housef_apply_l(A, v, beta, m - 1, m, m - 2, n - 1, 1);
					housef_apply_r(A, v, beta, 0, m, m - 1, m, 1);
					if (U) {
						housef_apply_l(U, v, beta, m - 1, m, 0, n - 1, 1);
					}
				}
			}
			for (k = q; k < m; k++) {
				if (fabsf(at(A, k + 1, k)) <=
					tol * (fabsf(at(A, k, k)) + fabsf(at(A, k + 1, k + 1)))) {
					at(A, k + 1, k) = 0;
				}
			}
			iter++;
		}
	}
	if (U) {
		matrixf_transpose(U);
	}
	// Trangularize all 2-by-2 diagonal blocks in A that have real eigenvalues,
	// and transform the blocks with complex eigenvalues so that the real part
	// of the eigenvalues appears on the main diagonal. 
	for (k = 0; k < n - 1; k++) {
		if (at(A, k + 1, k) != 0) {
			sine = 0;
			x = 0.5f * (at(A, k, k) - at(A, k + 1, k + 1));
			y = x * x + at(A, k + 1, k) * at(A, k, k + 1);
			if (y >= 0) { // real eigenvalues
				r = at(A, k, k + 1);
				s = at(A, k, k) - at(A, k + 1, k + 1) - x + sqrtf(y);
				t = hypotf(r, s);
				cosine = r / t;
				sine = s / t;
			}
			else if (at(A, k, k) != at(A, k + 1, k + 1)) {
				r = (at(A, k + 1, k) + at(A, k, k + 1)) /
					(at(A, k + 1, k + 1) - at(A, k, k));
				t = r - sqrtf(r * r + 1);
				cosine = 1.0f / sqrtf(t * t + 1);
				sine = cosine * t;
			}
			if (sine != 0) {
				colXk = &at(A, 0, k);
				colXk1 = &at(A, 0, k + 1);
				for (i = 0; i <= k + 1; i++) {
					Xk = colXk[i];
					Xk1 = colXk1[i];
					colXk[i] = Xk * cosine - Xk1 * sine;
					colXk1[i] = Xk * sine + Xk1 * cosine;
				}
				for (j = k; j < n; j++) {
					Xk = at(A, k, j);
					Xk1 = at(A, k + 1, j);
					at(A, k, j) = Xk * cosine - Xk1 * sine;
					at(A, k + 1, j) = Xk * sine + Xk1 * cosine;
				}
				if (U) {
					colXk = &at(U, 0, k);
					colXk1 = &at(U, 0, k + 1);
					for (i = 0; i < n; i++) {
						Xk = colXk[i];
						Xk1 = colXk1[i];
						colXk[i] = Xk * cosine - Xk1 * sine;
						colXk1[i] = Xk * sine + Xk1 * cosine;
					}
				}
			}
			if (y >= 0) {
				at(A, k + 1, k) = 0;
			}
			else {
				Xk = 0.5f * (at(A, k, k) + at(A, k + 1, k + 1));
				at(A, k, k) = Xk;
				at(A, k + 1, k + 1) = Xk;
			}
		}
		for (i = k + 2; i < n; i++) {
			at(A, i, k) = 0;
		}
	}
	return iter == iter_max ? -2 : iter;
}

int matrixf_get_eigenvectors(Matrixf* T, Matrixf* U,
	Matrixf* V, Matrixf* W, int pseudo, float* work)
{
	int i, j, k, h;
	const int n = T->rows;
	float norm, lamre, lamim, g;
	Matrixf C = { 0, 0, work }, d = { 0 };

	if (V) {
		if (V->rows != n || V->cols != n) {
			return -1;
		}
		for (i = 0; i < n * n; i++) {
			V->data[i] = !(i % (n + 1));
		}
	}
	if (W) {
		if (W->rows != n || W->cols != n) {
			return -1;
		}
		for (i = 0; i < n * n; i++) {
			W->data[i] = !(i % (n + 1));
		}
	}
	k = 0;
	while (k < n) {
		if (k == n - 1 || at(T, k + 1, k) == 0) { // real eigenvalue
			lamre = at(T, k, k) + epsf(at(T, k, k));
			if (V && k > 0) { // right eigenvector
				C.rows = C.cols = k;
				matrixf_init(&d, k, 1, &at(V, 0, k), 0);
				for (j = 0; j < k; j++) {
					for (i = 0; i < k; i++) {
						at(&C, i, j) = at(T, i, j);
					}
					at(&C, j, j) -= lamre;
					at(&d, j, 0) = -at(T, j, k);
				}
				if (matrixf_solve_lu(&C, &d)) {
					return -2;
				}
				norm = normf(&at(V, 0, k), n, 1);
				for (j = 0; j < n; j++) {
					at(V, j, k) /= norm;
				}
			}
			if (W && k < n - 1) { // left eigenvector
				h = n - 1 - k;
				C.rows = C.cols = h;
				matrixf_init(&d, h, 1, &at(W, k + 1, k), 0);
				for (j = 0; j < h; j++) {
					for (i = 0; i < h; i++) {
						at(&C, j, i) = at(T, k + 1 + i, k + 1 + j);
					}
					at(&C, j, j) -= lamre;
					at(&d, j, 0) = -at(T, k, k + 1 + j);
				}
				if (matrixf_solve_lu(&C, &d)) {
					return -2;
				}
				norm = normf(&at(W, 0, k), n, 1);
				for (j = 0; j < n; j++) {
					at(W, j, k) /= norm;
				}
			}
			k += 1;
		}
		else { // complex conjugate pair of eigenvalues
			// It is assumed that the 2-by-2 blocks have been transformed so 
			// that the real part of the eigenvalues appears on the diagonal
			lamre = at(T, k, k);
			lamim = sqrtf(-at(T, k + 1, k) * at(T, k, k + 1));
			if (pseudo) { // pseudo-eigenvectors
				g = at(T, k + 1, k) / lamim;
				if (V) { // right eigenvectors
					if (k > 0) {
						C.rows = C.cols = 2 * k;
						matrixf_init(&d, 2 * k, 1, &at(V, 0, k), 0);
						for (j = 0; j < k; j++) {
							for (i = 0; i < k; i++) {
								at(&C, i, j) = at(&C, k + i, k + j) = at(T, i, j);
								at(&C, i, k + j) = at(&C, k + i, j) = 0;
							}
							at(&C, j, j) -= at(T, k, k);
							at(&C, k + j, k + j) -= at(T, k + 1, k + 1);
							at(&C, j, k + j) = +lamim;
							at(&C, k + j, j) = -lamim;
							at(&d, j, 0) = -at(T, j, k) - at(T, k + 1, k) * at(T, j, k + 1);
							at(&d, k + j, 0) = -lamim * at(T, j, k) + g * at(T, j, k + 1);
						}
						if (matrixf_solve_lu(&C, &d)) {
							return -2;
						}
						for (j = k - 1; j >= 0; j--) {
							at(V, j, k + 1) = d.data[k + j];
							d.data[k + j] = 0;
						}
					}
					at(V, k, k) = 1;
					at(V, k + 1, k) = at(T, k + 1, k);
					at(V, k, k + 1) = lamim;
					at(V, k + 1, k + 1) = -g;
					norm = normf(&at(V, 0, k), 2 * n, 1);
					for (j = 0; j < n; j++) {
						at(V, j, k) /= norm;
						at(V, j, k + 1) /= norm;
					}
				}
				if (W) { // left eigenvectors
					if (k < n - 2) {
						h = n - 2 - k;
						C.rows = C.cols = 2 * h;
						matrixf_init(&d, 2 * h, 1, &at(W, k + 2, k), 0);
						for (j = 0; j < h; j++) {
							for (i = 0; i < h; i++) {
								at(&C, j, i) = at(&C, h + j, h + i) =
									at(T, k + 2 + i, k + 2 + j);
								at(&C, i, h + j) = at(&C, h + i, j) = 0;
							}
							at(&C, j, j) -= at(T, k, k);
							at(&C, h + j, h + j) -= at(T, k + 1, k + 1);
							at(&C, h + j, j) = +lamim;
							at(&C, j, h + j) = -lamim;
							at(&d, j, 0) = g * at(T, k, k + 2 + j) - lamim * at(T, k + 1, k + 2 + j);
							at(&d, h + j, 0) = -at(T, k + 1, k) * at(T, k, k + 2 + j) - at(T, k + 1, k + 2 + j);
						}
						if (matrixf_solve_lu(&C, &d)) {
							return -2;
						}
						for (j = h - 1; j >= 0; j--) {
							at(W, k + 2 + j, k + 1) = at(W, j, k + 1);
							at(W, j, k + 1) = 0;
						}
					}
					at(W, k, k) = -g;
					at(W, k + 1, k) = lamim;
					at(W, k, k + 1) = at(T, k + 1, k);
					at(W, k + 1, k + 1) = 1;
					norm = normf(&at(W, 0, k), 2 * n, 1);
					for (j = 0; j < n; j++) {
						at(W, j, k) /= norm;
						at(W, j, k + 1) /= norm;
					}
				}
			}
			else { // complex eigenvectors
				if (V) { // right eigenvectors
					if (k > 0) {
						C.rows = C.cols = 2 * k;
						matrixf_init(&d, 2 * k, 1, &at(V, 0, k), 0);
						for (j = 0; j < k; j++) {
							for (i = 0; i < k; i++) {
								at(&C, i, j) = at(&C, k + i, k + j) = at(T, i, j);
								at(&C, i, k + j) = at(&C, k + i, j) = 0;
							}
							at(&C, j, j) -= lamre;
							at(&C, k + j, k + j) -= lamre;
							at(&C, k + j, j) = -lamim;
							at(&C, j, k + j) = +lamim;
							at(&d, j, 0) = -at(T, j, k + 1) * at(T, k + 1, k);
							at(&d, k + j, 0) = -at(T, j, k) * lamim;
						}
						if (matrixf_solve_lu(&C, &d)) {
							return -2;
						}
						for (j = k - 1; j >= 0; j--) {
							at(V, j, k + 1) = d.data[k + j];
							d.data[k + j] = 0;
						}
					}
					at(V, k, k) = at(V, k + 1, k + 1) = 0;
					at(V, k + 1, k) = at(T, k + 1, k);
					at(V, k, k + 1) = lamim;
					norm = normf(&at(V, 0, k), 2 * n, 1);
					for (j = 0; j < 2 * n; j++) {
						at(V, j, k) /= norm;
					}
				}
				if (W) { // left eigenvectors
					if (k < n - 2) {
						h = n - 2 - k;
						C.rows = C.cols = 2 * h;
						matrixf_init(&d, 2 * h, 1, &at(W, k + 2, k), 0);
						for (j = 0; j < h; j++) {
							for (i = 0; i < h; i++) {
								at(&C, j, i) = at(&C, h + j, h + i) =
									at(T, k + 2 + i, k + 2 + j);
								at(&C, i, h + j) = at(&C, h + i, j) = 0;
							}
							at(&C, j, j) -= lamre;
							at(&C, h + j, h + j) -= lamre;
							at(&C, h + j, j) = +lamim;
							at(&C, j, h + j) = -lamim;
							at(&d, j, 0) = -at(T, k + 1, k + 2 + j) * at(T, k, k + 1);
							at(&d, h + j, 0) = at(T, k, k + 2 + j) * lamim;
						}
						if (matrixf_solve_lu(&C, &d)) {
							return -2;
						}
						for (j = h - 1; j >= 0; j--) {
							at(W, k + 2 + j, k + 1) = at(W, j, k + 1);
							at(W, j, k + 1) = 0;
						}
					}
					at(W, k, k) = at(W, k + 1, k + 1) = 0;
					at(W, k + 1, k) = at(T, k, k + 1);
					at(W, k, k + 1) = -lamim;
					norm = normf(&at(W, 0, k), 2 * n, 1);
					for (j = 0; j < 2 * n; j++) {
						at(W, j, k) /= norm;
					}
				}
			}
			k += 2;
		}
	}
	if (V) {
		matrixf_multiply_inplace(V, U, 0, 0, 0, work);
	}
	if (W) {
		matrixf_multiply_inplace(W, U, 0, 0, 0, work);
	}
	return 0;
}

int matrixf_solve_tril(Matrixf* L, Matrixf* B, Matrixf* X, int unitri)
{
	int i, j, k;
	const int m = L->rows;
	const int n = L->cols;
	const int h = B->cols;
	const int q = m < n ? m : n;
	float Lij, Lii;
	float* colBi, * colBj;

	if (B->rows != m || X->rows != n || X->cols != h) {
		return -1;
	}
	matrixf_transpose(B);
	for (i = 0; i < q; i++) {
		colBi = &at(B, 0, i);
		for (j = 0; j < i; j++) {
			Lij = at(L, i, j);
			colBj = &at(B, 0, j);
			for (k = 0; k < h; k++) {
				colBi[k] -= Lij * colBj[k];
			}
		}
		if (!unitri) {
			Lii = at(L, i, i);
			if (Lii == 0) {
				return -2;
			}
			for (k = 0; k < h; k++) {
				colBi[k] /= Lii;
			}
		}
	}
	matrixf_transpose(B);
	if (m < n) {
		for (k = h - 1; k >= 0; k--) {
			for (i = n - 1; i >= 0; i--) {
				at(X, i, k) = (i < q) ? at(B, i, k) : 0;
			}
		}
	}
	else {
		for (k = 0; k < h; k++) {
			for (i = 0; i < n; i++) {
				at(X, i, k) = (i < q) ? at(B, i, k) : 0;
			}
		}
	}
	return 0;
}

int matrixf_solve_triu(Matrixf* U, Matrixf* B, Matrixf* X, int unitri)
{
	int i, j, k;
	const int m = U->rows;
	const int n = U->cols;
	const int h = B->cols;
	const int q = m < n ? m : n;
	float Uij, Uii;
	float* colBi, * colBj;

	if (B->rows != m || X->rows != n || X->cols != h) {
		return -1;
	}
	matrixf_transpose(B);
	for (i = q - 1; i >= 0; i--) {
		colBi = &at(B, 0, i);
		for (j = i + 1; j < q; j++) {
			Uij = at(U, i, j);
			colBj = &at(B, 0, j);
			for (k = 0; k < h; k++) {
				colBi[k] -= Uij * colBj[k];
			}
		}
		if (!unitri) {
			Uii = at(U, i, i);
			if (Uii == 0) {
				return -2;
			}
			for (k = 0; k < h; k++) {
				colBi[k] /= Uii;
			}
		}
	}
	matrixf_transpose(B);
	if (m < n) {
		for (k = h - 1; k >= 0; k--) {
			for (i = n - 1; i >= 0; i--) {
				at(X, i, k) = (i < q) ? at(B, i, k) : 0;
			}
		}
	}
	else {
		for (k = 0; k < h; k++) {
			for (i = 0; i < n; i++) {
				at(X, i, k) = (i < q) ? at(B, i, k) : 0;
			}
		}
	}
	return 0;
}

int matrixf_solve_chol(Matrixf* A, Matrixf* B)
{
	int i, j, k;
	const int n = A->rows;
	const int p = B->cols;
	float Aji, Aij, Aii;
	float* colBi, * colBj;

	if (B->rows != n) {
		return -1;
	}
	k = matrixf_decomp_chol(A);
	if (k) {
		return k;
	}
	matrixf_transpose(B);
	for (i = 0; i < n; i++) {
		colBi = &at(B, 0, i);
		for (j = 0; j < i; j++) {
			Aji = at(A, j, i);
			colBj = &at(B, 0, j);
			for (k = 0; k < p; k++) {
				colBi[k] -= Aji * colBj[k];
			}
		}
		Aii = at(A, i, i);
		for (k = 0; k < p; k++) {
			colBi[k] /= Aii;
		}
	}
	for (i = n - 1; i >= 0; i--) {
		colBi = &at(B, 0, i);
		for (j = i + 1; j < n; j++) {
			Aij = at(A, i, j);
			colBj = &at(B, 0, j);
			for (k = 0; k < p; k++) {
				colBi[k] -= Aij * colBj[k];
			}
		}
		Aii = at(A, i, i);
		for (k = 0; k < p; k++) {
			colBi[k] /= Aii;
		}
	}
	matrixf_transpose(B);
	return 0;
}

int matrixf_solve_lu(Matrixf* A, Matrixf* B)
{
	int i, j, k;
	const int n = A->rows;
	const int p = B->cols;
	float Aij, Aii;
	float* colBi, * colBj;

	if (matrixf_decomp_lu(A, B)) {
		return -1;
	}
	matrixf_transpose(B);
	for (i = 1; i < n; i++) {
		colBi = &at(B, 0, i);
		for (j = 0; j < i; j++) {
			Aij = at(A, i, j);
			colBj = &at(B, 0, j);
			for (k = 0; k < p; k++) {
				colBi[k] -= Aij * colBj[k];
			}
		}
	}
	for (i = n - 1; i >= 0; i--) {
		colBi = &at(B, 0, i);
		for (j = i + 1; j < n; j++) {
			Aij = at(A, i, j);
			colBj = &at(B, 0, j);
			for (k = 0; k < p; k++) {
				colBi[k] -= Aij * colBj[k];
			}
		}
		Aii = at(A, i, i);
		if (Aii == 0) {
			return -2;
		}
		for (k = 0; k < p; k++) {
			colBi[k] /= Aii;
		}
	}
	matrixf_transpose(B);
	return 0;
}

int matrixf_solve_lu_banded(Matrixf* A, Matrixf* B, int ubw)
{
	int i, j, k, p;
	const int n = A->rows;
	const int h = B->cols;
	float Bik, Aii;
	float* colBk;

	if (matrixf_decomp_lu_banded(A, ubw)) {
		return -1;
	}
	if (matrixf_unpack_lu_banded(A, B)) {
		return -1;
	}
	for (k = 0; k < h; k++) {
		for (i = n - 1; i >= 0; i--) {
			p = ubw + i + 2;
			if (p > n) {
				p = n;
			}
			Bik = at(B, i, k);
			colBk = &at(B, 0, k);
			for (j = i + 1; j < p; j++) {
				Bik -= at(A, i, j) * colBk[j];
			}
			Aii = at(A, i, i);
			if (Aii == 0) {
				return -2;
			}
			at(B, i, k) = Bik / Aii;
		}
	}
	return 0;
}

int matrixf_solve_qr(Matrixf* A, Matrixf* B, Matrixf* X)
{
	const int m = A->rows;
	const int n = A->cols;
	const int p = B->cols;

	if (B->rows != m || X->rows != n || X->cols != p) {
		return -1;
	}
	if (m < n) {
		matrixf_transpose(A);
		matrixf_decomp_qr(A, 0, 0, 0);
		matrixf_transpose(A);
		if (matrixf_solve_tril(A, B, X, 0)) {
			return -2;
		}
		matrixf_transpose(A);
		matrixf_unpack_house(A, X, 0, 0);
	}
	else {
		matrixf_decomp_qr(A, 0, 0, B);
		if (matrixf_solve_triu(A, B, X, 0)) {
			return -2;
		}
	}
	return 0;
}

int matrixf_solve_qrp(Matrixf* A, Matrixf* B, Matrixf* X, float tol, float* work)
{
	int i, j, k, rank;
	const int m = A->rows;
	const int n = A->cols;
	const int h = B->cols;
	const int p = m < n ? m : n;
	float Aij, Aii;
	float* colBi, * colBj;
	Matrixf perm = { 1, n, work };

	if (B->rows != m || X->rows != n || X->cols != h) {
		return -1;
	}
	matrixf_decomp_qr(A, 0, &perm, B);
	if (tol < 0) {
		tol = (m > n ? m : n) * epsf(at(A, 0, 0));
	}
	j = 0;
	while (j < p && fabsf(at(A, j, j)) > tol) {
		j++;
	}
	rank = j;
	matrixf_transpose(B);
	for (i = rank - 1; i >= 0; i--) {
		colBi = &at(B, 0, i);
		for (j = i + 1; j < rank; j++) {
			Aij = at(A, i, j);
			colBj = &at(B, 0, j);
			for (k = 0; k < h; k++) {
				colBi[k] -= Aij * colBj[k];
			}
		}
		Aii = at(A, i, i);
		for (k = 0; k < h; k++) {
			colBi[k] /= Aii;
		}
	}
	matrixf_transpose(B);
	if (m < n) {
		for (k = h - 1; k >= 0; k--) {
			for (i = n - 1; i >= 0; i--) {
				at(X, i, k) = (i < rank) ? at(B, i, k) : 0;
			}
		}
	}
	else {
		for (k = 0; k < h; k++) {
			for (i = 0; i < n; i++) {
				at(X, i, k) = (i < rank) ? at(B, i, k) : 0;
			}
		}
	}
	matrixf_permute(X, &perm, 1, 0);
	return 0;
}

int matrixf_solve_cod(Matrixf* A, Matrixf* B, Matrixf* X, float tol, float* work)
{
	int i, j, k, rank;
	const int m = A->rows;
	const int n = A->cols;
	const int h = B->cols;
	const int p = m < n ? m : n;
	const int q = m > n ? m : n;
	float Aji, Aii;
	float* colBi, * colBj;
	Matrixf perm = { 1, n, work };

	if (B->rows != m || X->rows != n || X->cols != h) {
		return -1;
	}
	matrixf_decomp_qr(A, 0, &perm, B);
	if (tol < 0) {
		tol = q * epsf(at(A, 0, 0));
	}
	j = 0;
	while (j < p && fabsf(at(A, j, j)) > tol) {
		for (i = j + 1; i < m; i++) {
			at(A, i, j) = 0;
		}
		j++;
	}
	rank = j;
	matrixf_transpose(A);
	A->cols = rank;
	matrixf_decomp_qr(A, 0, 0, 0);
	matrixf_transpose(B);
	for (i = 0; i < rank; i++) {
		colBi = &at(B, 0, i);
		for (j = 0; j < i; j++) {
			Aji = at(A, j, i);
			colBj = &at(B, 0, j);
			for (k = 0; k < h; k++) {
				colBi[k] -= Aji * colBj[k];
			}
		}
		Aii = at(A, i, i);
		for (k = 0; k < h; k++) {
			colBi[k] /= Aii;
		}
	}
	matrixf_transpose(B);
	if (m < n) {
		for (k = h - 1; k >= 0; k--) {
			for (i = n - 1; i >= 0; i--) {
				at(X, i, k) = (i < rank) ? at(B, i, k) : 0;
			}
		}
	}
	else {
		for (k = 0; k < h; k++) {
			for (i = 0; i < n; i++) {
				at(X, i, k) = (i < rank) ? at(B, i, k) : 0;
			}
		}
	}
	matrixf_unpack_house(A, X, 0, 0);
	matrixf_permute(X, &perm, 1, 0);
	A->rows = m;
	A->cols = n;
	return 0;
}

int matrixf_pseudoinv(Matrixf* A, float tol, float* work)
{
	int i, j, iter;
	const int m = A->rows;
	const int n = A->cols;
	const int p = m < n ? m : n;
	const int q = m > n ? m : n;
	float sj, rj, * colXj;
	Matrixf V = { p, p, work + p };
	Matrixf* X = (m < n) ? A : &V;

	if (m < n) {
		matrixf_transpose(A);
	}
	iter = matrixf_decomp_svd_jacobi(A, 0, &V);
	if (tol < 0) {
		tol = q * epsf(normf(&at(A, 0, 0), q, 1));
	}
	for (j = 0; j < p; j++) {
		colXj = &at(A, 0, j);
		sj = normf(colXj, q, 1);
		if (sj > 0) {
			rj = sj > tol ? 1.0f / sj : 0;
			for (i = 0; i < q; i++) {
				colXj[i] /= sj;
			}
			colXj = &at(X, 0, j);
			for (i = 0; i < n; i++) {
				colXj[i] *= rj;
			}
		}
	}
	matrixf_transpose(A);
	matrixf_multiply_inplace(A, &V, 0, 0, 0, work);
	if (m < n) {
		matrixf_transpose(A);
	}
	return iter;
}

#ifndef DETECTUM_EXP_PADE_ORDER
#define DETECTUM_EXP_PADE_ORDER (4)
#endif
int matrixf_exp(Matrixf* A, float* work)
{
	int i, j, k, z, s;
	const int n = A->rows;
	const int q = DETECTUM_EXP_PADE_ORDER;
	float c, p, t;
	Matrixf X = { n, n, work + n };
	Matrixf N = { n, n, work + n + n * n };
	Matrixf D = { n, n, work + n + n * n * 2 };

	if (n != A->cols) {
		return -1;
	}
	for (p = 0, i = 0; i < n; i++) {
		for (t = 0, j = 0; j < n; j++) {
			t += fabsf(at(A, i, j));
		}
		if (t > p) {
			p = t;
		}
	}
	z = 1 + (int)floorf(log2f(p));
	if (z < 0) {
		z = 0;
	}
	s = 1 << z;
	for (i = 0; i < n * n; i++) {
		t = !(i % (n + 1));
		X.data[i] = t;
		N.data[i] = t;
		D.data[i] = t;
		A->data[i] /= s;
	}
	for (c = 1, k = 1; k <= q; k++) {
		c *= (float)(q - k + 1) / ((2 * q - k + 1) * k);
		matrixf_multiply_inplace(&X, A, 0, 0, 0, work);
		t = (k % 2) ? -1.0f : 1.0f;
		for (i = 0; i < n * n; i++) {
			N.data[i] += c * X.data[i];
			D.data[i] += c * X.data[i] * t;
		}
	}
	for (i = 0; i < n * n; i++) {
		A->data[i] = N.data[i];
	}
	if (matrixf_solve_lu(&D, A)) {
		return -2;
	}
	for (k = 0; k < z; k++) {
		for (i = 0; i < n * n; i++) {
			N.data[i] = A->data[i];
		}
		matrixf_multiply(&N, &N, A, 1, 0, 0, 0);
	}
	return 0;
}

#ifndef DETECTUM_LOG_ISS_THR
#define DETECTUM_LOG_ISS_THR (0.5f)
#endif
#ifndef DETECTUM_LOG_NTERMS
#define DETECTUM_LOG_NTERMS (5)
#endif
int matrixf_log(Matrixf* A, float* work)
{
	int i, k, f, s = 1;
	const int n = A->rows;
	const int nterms = DETECTUM_LOG_NTERMS;
	const float thr = DETECTUM_LOG_ISS_THR;
	float tmp, norm1;
	Matrixf U = { n, n, work + n };
	Matrixf N = { n, n, work + n + n * n };
	Matrixf D = { n, n, work + n + n * n * 2 };

	f = matrixf_decomp_schur(A, &U);
	if (f < 0) {
		return f;
	}
	do {
		f = matrixf_sqrt_quasitriu(A);
		if (f < 0) {
			return f;
		}
		else if (f == 1) {
			return -3;
		}
		s <<= 1;
		for (norm1 = 0, k = 0; k < n; k++) {
			for (tmp = 0, i = 0; i < n; i++) {
				tmp += fabsf(at(A, i, k) - (i == k));
			}
			if (tmp > norm1) {
				norm1 = tmp;
			}
		}
	} while (norm1 > thr);
	matrixf_transpose(A);
	for (k = 0; k < n * n; k++) {
		f = !(k % (n + 1));
		D.data[k] = f + A->data[k];
		N.data[k] = f - A->data[k];
	}
	if (matrixf_solve_lu(&D, &N) < 0) {
		return -4;
	}
	matrixf_transpose(&N);
	for (k = 0; k < n * n; k++) {
		A->data[k] = N.data[k];
	}
	matrixf_multiply(A, A, &D, 1, 0, 0, 0);
	for (i = 1; i < nterms; i++) {
		matrixf_multiply_inplace(&N, 0, &D, 0, 0, work);
		for (k = 0; k < n * n; k++) {
			A->data[k] += N.data[k] / (2 * i + 1);
		}
	}
	for (k = 0; k < n * n; k++) {
		A->data[k] *= -2 * s;
	}
	matrixf_multiply_inplace(A, &U, &U, 0, 1, work);
	return 0;
}

int matrixf_sqrt_quasitriu(Matrixf* T)
{
	int i, j = 0, r, k1, kj = 0, kr, kr1, sj, sr, singular = 0;
	const int n = T->rows;
	float T00, T10, T01, t;
	float* k = T->data, C_data[16] = { 0 }, D_data[4] = { 0 };
	Matrixf C = { 0, 0, C_data };
	Matrixf D = { 0, 1, D_data };

	while (kj < n) {
		sj = kj < n - 1 ? 1 + (at(T, kj + 1, kj) != 0) : 1;
		if (sj == 1) {
			T00 = at(T, kj, kj);
			if (T00 > 0) {
				at(T, kj, kj) = sqrtf(T00);
			}
			else if (T00 == 0) {
				singular = 1;
			}
			else {
				return -3;
			}
		}
		else {
			T00 = at(T, kj, kj);
			T10 = at(T, kj + 1, kj);
			T01 = at(T, kj, kj + 1);
			t = sqrtf(0.5f * (T00 + sqrtf(T00 * T00 - T10 * T01)));
			at(T, kj, kj) = at(T, kj + 1, kj + 1) = t;
			at(T, kj + 1, kj) = 0.5f * T10 / t;
			at(T, kj, kj + 1) = 0.5f * T01 / t;
		}
		if (j > 1) {
			k[j] = (float)kj;
		}
		else if (j == 1) {
			k1 = kj;
		}
		for (r = j - 1; r >= 0; r--) {
			if (r == 0) {
				kr = 0;
				kr1 = k1;
			}
			else if (r == 1) {
				kr = k1;
				kr1 = (int)k[2];
			}
			else {
				kr = (int)k[r];
				kr1 = (int)k[r + 1];
			}
			sr = 1 + (at(T, kr + 1, kr) != 0);
			D.data[0] = at(T, kr, kj);
			if (sj + sr == 2) {
				C.data[0] = at(T, kr, kr) + at(T, kj, kj);
				for (i = kr1; i < kj; i++) {
					D.data[0] -= at(T, kr, i) * at(T, i, kj);
				}
				if (C.data[0] == 0) {
					return -4;
				}
				at(T, kr, kj) = D.data[0] / C.data[0];
			}
			else if (sj + sr == 3) {
				C.rows = C.cols = D.rows = 2;
				if (sr > sj) {
					D.data[1] = at(T, kr + 1, kj);
					for (i = kr1; i < kj; i++) {
						D.data[0] -= at(T, kr, i) * at(T, i, kj);
						D.data[1] -= at(T, kr + 1, i) * at(T, i, kj);
					}
					C.data[0] = at(T, kr, kr) + at(T, kj, kj);
					C.data[1] = at(T, kr + 1, kr);
					C.data[2] = at(T, kr, kr + 1);
					C.data[3] = at(T, kr + 1, kr + 1) + at(T, kj, kj);
					if (matrixf_solve_lu(&C, &D)) {
						return -4;
					}
					at(T, kr, kj) = D.data[0];
					at(T, kr + 1, kj) = D.data[1];
				}
				else {
					D.data[1] = at(T, kr, kj + 1);
					for (i = kr1; i < kj; i++) {
						D.data[0] -= at(T, kr, i) * at(T, i, kj);
						D.data[1] -= at(T, kr, i) * at(T, i, kj + 1);
					}
					C.data[0] = at(T, kr, kr) + at(T, kj, kj);
					C.data[1] = at(T, kj, kj + 1);
					C.data[2] = at(T, kj + 1, kj);
					C.data[3] = at(T, kr, kr) + at(T, kj + 1, kj + 1);
					if (matrixf_solve_lu(&C, &D)) {
						return -4;
					}
					at(T, kr, kj) = D.data[0];
					at(T, kr, kj + 1) = D.data[1];
				}
			}
			else if (sj + sr == 4) {
				C.rows = C.cols = D.rows = 4;
				D.data[1] = at(T, kr + 1, kj);
				D.data[2] = at(T, kr, kj + 1);
				D.data[3] = at(T, kr + 1, kj + 1);
				for (i = kr1; i < kj; i++) {
					D.data[0] -= at(T, kr, i) * at(T, i, kj);
					D.data[1] -= at(T, kr + 1, i) * at(T, i, kj);
					D.data[2] -= at(T, kr, i) * at(T, i, kj + 1);
					D.data[3] -= at(T, kr + 1, i) * at(T, i, kj + 1);
				}
				C.data[0] = C.data[10] = at(T, kr, kr) + at(T, kj, kj);
				C.data[1] = C.data[11] = at(T, kr + 1, kr);
				C.data[4] = C.data[14] = at(T, kr, kr + 1);
				C.data[5] = C.data[15] = at(T, kr + 1, kr + 1) + at(T, kj, kj);
				C.data[3] = C.data[6] = C.data[9] = C.data[12] = 0;
				C.data[2] = C.data[7] = at(T, kj, kj + 1);
				C.data[8] = C.data[13] = at(T, kj + 1, kj);
				if (matrixf_solve_lu(&C, &D)) {
					return -4;
				}
				at(T, kr, kj) = D.data[0];
				at(T, kr + 1, kj) = D.data[1];
				at(T, kr, kj + 1) = D.data[2];
				at(T, kr + 1, kj + 1) = D.data[3];
			}
		}
		kj += sj;
		j++;
	}
	for (i = 2; i < n; i++) {
		k[i] = 0;
	}
	return singular;
}

int matrixf_sqrt(Matrixf* A, float* work)
{
	int f;
	const int n = A->rows;
	Matrixf U = { n, n, work + n };

	f = matrixf_decomp_schur(A, &U);
	if (f < 0) {
		return f;
	}
	f = matrixf_sqrt_quasitriu(A);
	if (f < 0) {
		return f;
	}
	matrixf_multiply_inplace(A, &U, &U, 0, 1, work);
	return f;
}

int matrixf_multiply(Matrixf* A, Matrixf* B, Matrixf* C,
	float alpha, float beta, int transA, int transB)
{
	int i, j, k;
	const int m = C->rows;
	const int n = C->cols;
	const int p = transA ? A->rows : A->cols;
	const int q = transB ? B->cols : B->rows;
	const int r = transB ? B->rows : B->cols;
	const int f = transA && (A == B) ? !transB : transB;
	float b, * colAk, * colCj;

	for (i = 0; i < m * n; i++) {
		C->data[i] *= beta;
	}
	if (alpha == 0) {
		return 0;
	}
	if (transA) {
		matrixf_transpose(A);
	}
	if (m != A->rows || p != q || r != n) {
		return -1;
	}
	for (j = 0; j < n; j++) {
		colCj = &at(C, 0, j);
		for (k = 0; k < p; k++) {
			b = f ? at(B, j, k) : at(B, k, j);
			b *= alpha;
			colAk = &at(A, 0, k);
			for (i = 0; i < m; i++) {
				colCj[i] += colAk[i] * b;
			}
		}
	}
	if (transA) {
		matrixf_transpose(A);
	}
	return 0;
}

int matrixf_multiply_inplace(Matrixf* A, Matrixf* L, Matrixf* R,
	int transL, int transR, float* work)
{
	int i, j, k;
	const int m = A->rows;
	const int n = A->cols;
	float* colA, * colX;

	if (L) {
		if (m != L->rows || m != L->cols) {
			return -1;
		}
		if (transL) { // A = L' * A
			for (j = 0; j < n; j++) {
				colA = &at(A, 0, j);
				for (i = 0; i < m; i++) {
					work[i] = colA[i];
					colA[i] = 0;
				}
				for (i = 0; i < m; i++) {
					colX = &at(L, 0, i);
					for (k = 0; k < m; k++) {
						colA[i] += work[k] * colX[k];
					}
				}
			}
		}
		else { // A = L * A
			for (j = 0; j < n; j++) {
				colA = &at(A, 0, j);
				for (i = 0; i < m; i++) {
					work[i] = colA[i];
					colA[i] = 0;
				}
				for (k = 0; k < m; k++) {
					colX = &at(L, 0, k);
					for (i = 0; i < m; i++) {
						colA[i] += work[k] * colX[i];
					}
				}
			}
		}
	}
	if (R) {
		if (n != R->rows || n != R->cols) {
			return -1;
		}
		matrixf_transpose(A);
		if (transR) { // A = A * R'
			for (j = 0; j < m; j++) {
				colA = &at(A, 0, j);
				for (i = 0; i < n; i++) {
					work[i] = colA[i];
					colA[i] = 0;
				}
				for (k = 0; k < n; k++) {
					colX = &at(R, 0, k);
					for (i = 0; i < n; i++) {
						colA[i] += work[k] * colX[i];
					}
				}
			}
		}
		else { // A = A * R
			for (j = 0; j < m; j++) {
				colA = &at(A, 0, j);
				for (i = 0; i < n; i++) {
					work[i] = colA[i];
					colA[i] = 0;
				}
				for (i = 0; i < n; i++) {
					colX = &at(R, 0, i);
					for (k = 0; k < n; k++) {
						colA[i] += work[k] * colX[k];
					}
				}
			}
		}
		matrixf_transpose(A);
	}
	return 0;
}