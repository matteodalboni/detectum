#include "detego.h"

// This function swaps the columns i and j of the matrix A.
static void swap_columns(Matrixf* A, const int i, const int j)
{
	int k;
	const int m = A->size[0];
	float* x = &at(A, 0, i);
	float* y = &at(A, 0, j);
	float xk;

	for (k = 0; k < m; k++) {
		xk = x[k];
		x[k] = y[k];
		y[k] = xk;
	}
}

// Householder transformation X(i0:iend,j0:jend) = H*X(i0:iend,j0:jend),
// where H = I - beta*v*v'; stride is the increment of v.
static void householder_hx(Matrixf* X, float* v, const float beta,
	const int i0, const int iend, const int j0, const int jend, const int stride)
{
	int i, j;
	float h;

	for (j = j0; j <= jend; j++) {
		h = at(X, i0, j) * v[0];
		for (i = i0 + 1; i <= iend; i++) {
			h += at(X, i, j) * v[(i - i0) * stride];
		}
		at(X, i0, j) -= beta * h;
		for (i = i0 + 1; i <= iend; i++) {
			at(X, i, j) -= beta * h * v[(i - i0) * stride];
		}
	}
}

// Householder transformation X(i0:iend,j0:jend) = X(i0:iend,j0:jend)*H,
// where H = I - beta*v*v'; stride is the increment of v.
static void householder_xh(Matrixf* X, float* v, const float beta,
	const int i0, const int iend, const int j0, const int jend, const int stride)
{
	int i, j;
	float h;

	for (i = i0; i <= iend; i++) {
		h = at(X, i, j0) * v[0];
		for (j = j0 + 1; j <= jend; j++) {
			h += at(X, i, j) * v[(j - j0) * stride];
		}
		at(X, i, j0) -= beta * h;
		for (j = j0 + 1; j <= jend; j++) {
			at(X, i, j) -= beta * h * v[(j - j0) * stride];
		}
	}
}

void matrixf_init(Matrixf* A, int rows, int cols, float* data, const int ordmem)
{
	A->data = data;
	A->size[ordmem != 0] = rows;
	A->size[ordmem == 0] = cols;
	if (ordmem) {
		matrixf_transpose(A);
	}
}

int matrixf_permute(Matrixf* A, Matrixf* p, const int reverse)
{
	int i, j, k, t;
	const int m = (reverse) ? p->size[1] : p->size[0];
	const int n = (reverse) ? p->size[0] : p->size[1];
	const int h = m > n ? m : n;
	float* x = p->data;

	if ((A->size[0] != m || n != 1) &&
		(A->size[1] != n || m != 1)) {
		return -1;
	}
	if (reverse && h > 1) {
		p->size[0] = m;
		p->size[1] = n;
		for (k = 0; k < h; k++) {
			i = (int)x[k];
			if (i >= 0) {
				j = k;
				do {
					t = (int)x[i];
					x[i] = -(float)(j + 1);
					j = i;
					i = t;
				} while (j != k);
			}
		}
		for (k = 0; k < h; k++) {
			x[k] = -x[k] - 1;
		}
	}
	if (n == 1) {
		matrixf_transpose(A);
	}
	for (i = 0; i < h - 1; i++) {
		j = (int)x[i];
		while (j < i) {
			j = (int)x[j];
		}
		if (i != j) {
			swap_columns(A, i, j);
		}
	}
	if (n == 1) {
		matrixf_transpose(A);
	}
	return 0;
}

void matrixf_transpose(Matrixf* A)
{
	int i, j;
	float t, * d = A->data;
	const int r = A->size[0];
	const int c = A->size[1];
	const int n = r * c - 1;

	for (j = i = 1; i < n; j = ++i) {
		do {
			j = j * r - n * (j / c);
		} while (j < i);
		t = d[i];
		d[i] = d[j];
		d[j] = t;
	}
	A->size[0] = c;
	A->size[1] = r;
}

int matrixf_decomp_chol(Matrixf* A)
{
	int i, j, k;
	const int n = A->size[0];
	float v, r = 0;

	if (A->size[1] != n) {
		return -1;
	}
	for (j = 0; j < n; j++) {
		for (i = j; i < n; i++) {
			for (v = at(A, j, i), k = 0; k < j; k++) {
				v -= at(A, k, i) * at(A, k, j);
			}
			if (i == j) {
				if (v > 0) {
					r = sqrtf(v);
				}
				else {
					return 1;
				}
			}
			at(A, j, i) = v / r;
		}
	}
	return 0;
}

int matrixf_decomp_ltl(Matrixf* A)
{
	int i, j, k, m;
	const int n = A->size[0];
	float s, Ljj, P1, A0 = A->data[0], * h = A->data;

	if (A->size[1] != n) {
		return -1;
	}
	for (j = 1; j < n; j++) {
		at(A, 0, j) = (float)j;
	}
	for (j = 0; j < n; j++) {
		if (j) {
			h[0] = (j == 1) ? at(A, 0, 1) : at(A, 0, 1) * at(A, j, 1);
			for (k = 1; k < j; k++) {
				Ljj = (j == k + 1) ? 1.0f : at(A, j, k + 1);
				h[k] = (k > 1) ? at(A, k - 1, k) * at(A, j, k - 1) : 0;
				h[k] += at(A, k, k) * at(A, j, k) + at(A, k, k + 1) * Ljj;
			}
			for (s = 0, k = 1; k < j; k++) {
				s += at(A, j, k) * h[k];
			}
			h[j] = at(A, j, j) - s;
			at(A, j, j) = (j > 1) ? h[j] - at(A, j - 1, j) * at(A, j, j - 1) : h[j];
			for (i = j + 1; i < n; i++) {
				for (s = 0, k = 1; k <= j; k++) {
					s += at(A, i, k) * h[k];
				}
				h[i] = at(A, j, i) - s;
			}
		}
		if (j < n - 1) {
			for (m = j + 1, k = j + 1; k < n; k++) {
				if (fabsf(h[k]) > fabsf(h[m])) {
					m = k;
				}
			}
			for (k = 0; k < n; k++) {
				s = at(A, m, k);
				at(A, m, k) = at(A, j + 1, k);
				at(A, j + 1, k) = s;
			}
			swap_columns(A, m, j + 1);
			if (!j) {
				P1 = at(A, 0, 1);
			}
			at(A, j, j + 1) = h[j + 1];
			if (j < n - 2) {
				for (i = j + 2; i < n; i++) {
					at(A, i, j + 1) = h[m] ? h[i] / h[j + 1] : 0;
				}
			}
		}
	}
	A->data[0] = A0;
	if (n > 1) {
		A->data[1] = P1;
	}
	for (j = 2; j < n; j++) {
		at(A, j, 0) = at(A, 0, j);
		for (i = 0; i < j - 1; i++) {
			at(A, i, j) = 0;
		}
	}
	return 0;
}

int matrixf_decomp_lu(Matrixf* A, Matrixf* P)
{
	int i, j, k;
	const int n = A->size[0];
	float a, b;

	if (A->size[1] != n || P->size[0] != n) {
		return -1;
	}
	matrixf_transpose(A);
	matrixf_transpose(P);
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
		swap_columns(A, k, i);
		swap_columns(P, k, i);
		for (k = i + 1; k < n; k++) {
			a = at(A, i, i);
			if (a != 0) {
				at(A, i, k) /= a;
			}
			for (j = i + 1; j < n; j++) {
				at(A, j, k) -= at(A, j, i) * at(A, i, k);
			}
		}
	}
	matrixf_transpose(A);
	matrixf_transpose(P);
	return 0;
}

int matrixf_decomp_lu_banded(Matrixf* A, const int ubw)
{
	int i, j, p, piv;
	const int n = A->size[0];
	float tau, t;

	if (A->size[1] != n || ubw < 0) {
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
	const int n = A->size[0];
	const int p = B->size[1];
	float t, tau;

	if (B->size[0] != n) {
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
	int i, km = 0, k = 0, r = 0, q = 0;
	const int m = A->size[0];
	const int n = A->size[1];
	const int rmax = m < n ? m - 1 : n - 1;
	float alpha, beta, s, t, c, cm = 1, * v, v0;
	Matrixf A_econ = { { n, n }, A->data };

	if (Q) {
		q = Q->size[1];
		if (Q->size[0] != m || (q != m && q != n) || (q == n && m < n)) {
			return -1;
		}
		for (i = 0; i < m * q; i++) {
			Q->data[i] = !(i % (m + 1));
		}
	}
	if (P) {
		if ((P->size[0] != 1 && P->size[0] != n) || P->size[1] != n) {
			return -1;
		}
		for (cm = 0, km = 0, k = 0; k < n; k++) {
			for (c = 0, i = 0; i < m; i++) {
				c += at(A, i, k) * at(A, i, k);
			}
			if (c > cm) {
				cm = c;
				km = k;
			}
			P->data[k] = (float)k;
		}
	}
	if (B && B->size[0] != m) {
		return -1;
	}
	while (cm > 0 && r <= rmax) {
		if (P) {
			swap_columns(A, r, km);
			t = P->data[r];
			P->data[r] = P->data[km];
			P->data[km] = t;
		}
		i = r + 1;
		while (i < m && at(A, i, r) == 0) {
			i++;
		}
		if (i < m) {
			alpha = normf(&at(A, r, r), m - r, 1);
			s = at(A, r, r) >= 0 ? 1.0f : -1.0f;
			t = at(A, r, r) + s * alpha;
			at(A, r, r) = -s * alpha;
			for (i = r + 1; i < m; i++) {
				at(A, i, r) /= t;
			}
			beta = t * s / alpha;
			v0 = at(A, r, r);
			v = &at(A, r, r);
			v[0] = 1;
			householder_hx(A, v, beta, r, m - 1, r + 1, n - 1, 1);
			if (B) {
				householder_hx(B, v, beta, r, m - 1, 0, B->size[1] - 1, 1);
			}
			if (Q && q == m) {
				householder_xh(Q, v, beta, 0, m - 1, r, m - 1, 1);
				for (i = r + 1; i < m; i++) {
					at(A, i, r) = 0;
				}
			}
			at(A, r, r) = v0;
		}
		if (P && r + 1 <= rmax) {
			for (cm = 0, km = 0, k = r + 1; k < n; k++) {
				for (c = 0, i = r + 1; i < m; i++) {
					c += at(A, i, k) * at(A, i, k);
				}
				if (c > cm) {
					cm = c;
					km = k;
				}
			}
		}
		r++;
	}
	if (Q && q < m) {
		matrixf_unpack_householder_bwd(A, Q, 0);
		for (k = 0; k < n; k++) {
			for (i = 0; i < n; i++) {
				at(&A_econ, i, k) = (i <= k) ? at(A, i, k) : 0;
			}
		}
		A->size[0] = n;
	}
	if (P && P->size[0] == n) {
		for (i = n; i < n * n; i++) {
			P->data[i] = 0;
		}
		for (i = n - 1; i >= 0; i--) {
			k = (int)P->data[i];
			at(P, i, 0) = 0;
			at(P, k, i) = 1;
		}
	}
	return 0;
}

int matrixf_unpack_householder_fwd(Matrixf* A, Matrixf* B, const int s)
{
	int i, k;
	const int m = A->size[0];
	const int n = A->size[1];
	const int p = B->size[1];
	const int kmax = m - 1 < n ? m - 2 : n - 1;
	float alpha, * v, v0;

	if (B->size[0] != m || s < 0) {
		return -1;
	}
	for (k = 0; k <= kmax - s; k++) {
		for (alpha = 1, i = k + 1 + s; i < m; i++) {
			alpha += at(A, i, k) * at(A, i, k);
		}
		if (alpha > 1) {
			v0 = at(A, k + s, k);
			v = &at(A, k + s, k);
			v[0] = 1;
			householder_hx(B, v, 2.0f / alpha, k + s, m - 1, 0, p - 1, 1);
			at(A, k + s, k) = v0;
		}
	}
	return 0;
}

int matrixf_unpack_householder_bwd(Matrixf* A, Matrixf* B, const int s)
{
	int i, k;
	const int m = A->size[0];
	const int n = A->size[1];
	const int p = B->size[1];
	const int kmax = m - 1 < n ? m - 2 : n - 1;
	float alpha, * v, v0;

	if (B->size[0] != m || s < 0) {
		return -1;
	}
	for (k = kmax - s; k >= 0; k--) {
		for (alpha = 1, i = k + 1 + s; i < m; i++) {
			alpha += at(A, i, k) * at(A, i, k);
		}
		if (alpha > 1) {
			v0 = at(A, k + s, k);
			v = &at(A, k + s, k);
			v[0] = 1;
			householder_hx(B, v, 2.0f / alpha, k + s, m - 1, 0, p - 1, 1);
			at(A, k + s, k) = v0;
		}
	}
	return 0;
}

int matrixf_decomp_bidiag(Matrixf* A, Matrixf* U, Matrixf* V)
{
	int i, k, q = A->size[0];
	const int m = A->size[0];
	const int n = A->size[1];
	const int kmax = m - 1 < n ? m - 2 : n - 1;
	float alpha, beta, s, t, * v, v0;
	Matrixf A_econ = { { n, n }, A->data };

	if (m < n) {
		matrixf_transpose(A);
		k = matrixf_decomp_bidiag(A, V, U);
		matrixf_transpose(A);
		return k;
	}
	if ((U && (U->size[0] != m || (U->size[1] != m && U->size[1] != n))) ||
		(V && (V->size[0] != n || (V->size[1] != n)))) {
		return -1;
	}
	if (U) {
		q = U->size[1];
		for (i = 0; i < m * q; i++) {
			U->data[i] = !(i % (m + 1));
		}
	}
	if (V) {
		for (i = 0; i < n * n; i++) {
			V->data[i] = !(i % (n + 1));
		}
	}
	for (k = 0; k <= kmax; k++) {
		i = k + 1;
		while (i < m && at(A, i, k) == 0) {
			i++;
		}
		if (i < m) {
			alpha = normf(&at(A, k, k), m - k, 1);
			s = at(A, k, k) >= 0 ? 1.0f : -1.0f;
			t = at(A, k, k) + s * alpha;
			at(A, k, k) = -s * alpha;
			for (i = k + 1; i < m; i++) {
				at(A, i, k) /= t;
			}
			beta = t * s / alpha;
			v0 = at(A, k, k);
			v = &at(A, k, k);
			v[0] = 1;
			householder_hx(A, v, beta, k, m - 1, k + 1, n - 1, 1);
			if (U && q == m) {
				householder_xh(U, v, beta, 0, m - 1, k, m - 1, 1);
				for (i = k + 1; i < m; i++) {
					at(A, i, k) = 0;
				}
			}
			at(A, k, k) = v0;
		}
		i = k + 2;
		while (i < n && at(A, k, i) == 0) {
			i++;
		}
		if (i < n && k + 2 < n) {
			alpha = normf(&at(A, k, k + 1), n - k - 1, m);
			s = at(A, k, k + 1) >= 0 ? 1.0f : -1.0f;
			t = at(A, k, k + 1) + s * alpha;
			at(A, k, k + 1) = -s * alpha;
			for (i = k + 2; i < n; i++) {
				at(A, k, i) /= t;
			}
			beta = t * s / alpha;
			v0 = at(A, k, k + 1);
			v = &at(A, k, k + 1);
			v[0] = 1;
			householder_xh(A, v, beta, k + 1, m - 1, k + 1, n - 1, m);
			if (V) {
				householder_xh(V, v, beta, 0, n - 1, k + 1, n - 1, m);
				for (i = k + 2; i < n; i++) {
					at(A, k, i) = 0;
				}
			}
			at(A, k, k + 1) = v0;
		}
	}
	if (U && q < m) {
		matrixf_unpack_householder_bwd(A, U, 0);
		for (k = 0; k < n; k++) {
			for (i = 0; i < n; i++) {
				at(&A_econ, i, k) = (i <= k) ? at(A, i, k) : 0;
			}
		}
		A->size[0] = n;
	}
	return 0;
}

int matrixf_decomp_svd(Matrixf* A, Matrixf* U, Matrixf* V)
{
	const int m = A->size[0];
	const int n = A->size[1];
	int i, j, k, q, r = n - 1, sweep = 0;
	const int sweepmax = DETEGO_SVD_SWEEPMAX;
	const float tol = DETEGO_SVD_TOL;
	float small, norm1, tmp, cosine, sine, Xi, Xj;
	float c00, c01, c11, y, z, mu;
	float* s = A->data, * p = 0;
	Matrixf perm = { { 1, n }, p };

	if (m < n) {
		matrixf_transpose(A);
		k = matrixf_decomp_svd(A, V, U);
		matrixf_transpose(A);
		return k;
	}
	if (matrixf_decomp_bidiag(A, U, V)) {
		return -1;
	}
	for (j = 0; j < n; j++) {
		for (i = 0; i < m; i++) {
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
	while (r > 0 && sweep < sweepmax) {
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
				for (k = (r - 2 < 0 ? 0 : r - 2); k <= r; k++) {
					Xi = at(A, k, r - 1);
					Xj = at(A, k, r);
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
					for (k = (j - 1 < 0 ? 0 : j - 1); k <= j + 1; k++) {
						Xi = at(A, k, j);
						Xj = at(A, k, j + 1);
						at(A, k, j) = cosine * Xi - sine * Xj;
						at(A, k, j + 1) = sine * Xi + cosine * Xj;
					}
					if (V) {
						for (k = 0; k < n; k++) {
							Xi = at(V, k, j);
							Xj = at(V, k, j + 1);
							at(V, k, j) = cosine * Xi - sine * Xj;
							at(V, k, j + 1) = sine * Xi + cosine * Xj;
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
						for (k = 0; k < m; k++) {
							Xi = at(U, k, j);
							Xj = at(U, k, j + 1);
							at(U, k, j) = cosine * Xi - sine * Xj;
							at(U, k, j + 1) = sine * Xi + cosine * Xj;
						}
					}
					if (j < r - 1) {
						y = at(A, j, j + 1);
						z = at(A, j, j + 2);
					}
				}
				sweep++;
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
						for (k = 0; k < m; k++) {
							Xi = at(U, k, i);
							Xj = at(U, k, j);
							at(U, k, i) = cosine * Xi - sine * Xj;
							at(U, k, j) = sine * Xi + cosine * Xj;
						}
					}
				}
			}
			else {
				for (j = r - 1; j >= q; j--) {
					givensf(at(A, j, j), at(A, j, r), &cosine, &sine);
					for (k = (j - 1 > q ? j - 1 : q); k <= j; k++) {
						Xi = at(A, k, j);
						Xj = at(A, k, r);
						at(A, k, j) = cosine * Xi - sine * Xj;
						at(A, k, r) = sine * Xi + cosine * Xj;
					}
					if (V) {
						for (k = 0; k < n; k++) {
							Xi = at(V, k, j);
							Xj = at(V, k, r);
							at(V, k, j) = cosine * Xi - sine * Xj;
							at(V, k, r) = sine * Xi + cosine * Xj;
						}
					}
				}
			}
		}
	}
	if (n > 1) {
		p = A->data + A->size[0];
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
			k = U->size[1];
			U->size[1] = n;
			matrixf_permute(U, &perm, 0);
			U->size[1] = k;
		}
		if (V) {
			matrixf_permute(V, &perm, 0);
		}
		p[0] = 0;
		for (j = 1; j < n; j++) {
			p[j] = 0;
			at(A, j, j) = s[j];
			s[j] = 0;
		}
	}
	return sweep;
}

int matrixf_decomp_svd_jacobi(Matrixf* A, Matrixf* U, Matrixf* V)
{
	int i, j, k, count = 1, sweep = 0, sorted, orthog;
	float a, b, p, q, v, Xij, Xik, s, sine, cosine;
	const int m = A->size[0];
	const int n = A->size[1];
	const int sweepmax = DETEGO_SVD_JACOBI_SWEEPMAX;
	const float tol = DETEGO_SVD_JACOBI_TOL;

	if (m < n) {
		matrixf_transpose(A);
		k = matrixf_decomp_svd_jacobi(A, V, U);
		matrixf_transpose(A);
		return k;
	}
	if (U) {
		if (U->size[0] != m || (U->size[1] != m && U->size[1] != n)) {
			return -1;
		}
	}
	if (V) {
		if (V->size[0] != n || V->size[1] != n) {
			return -1;
		}
		for (i = 0; i < n * n; i++) {
			V->data[i] = !(i % (n + 1));
		}
	}
	while (count > 0 && sweep < sweepmax) {
		count = n * (n - 1) / 2;
		for (j = 0; j < n - 1; j++) {
			for (k = j + 1; k < n; k++) {
				a = b = p = 0;
				for (i = 0; i < m; i++) {
					a += at(A, i, j) * at(A, i, j);
					b += at(A, i, k) * at(A, i, k);
					p += at(A, i, j) * at(A, i, k);
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
						sine = sqrtf((v - q) / (2.0f * v)) * s;
						cosine = p / (2.0f * v * sine);
					}
					else {
						cosine = sqrtf((v + q) / (2.0f * v));
						sine = p / (2.0f * v * cosine);
					}
					for (i = 0; i < m; i++) {
						Xij = at(A, i, j);
						Xik = at(A, i, k);
						at(A, i, j) = +Xij * cosine + Xik * sine;
						at(A, i, k) = -Xij * sine + Xik * cosine;
					}
					if (V) {
						for (i = 0; i < n; i++) {
							Xij = at(V, i, j);
							Xik = at(V, i, k);
							at(V, i, j) = +Xij * cosine + Xik * sine;
							at(V, i, k) = -Xij * sine + Xik * cosine;
						}
					}
				}
			}
		}
		sweep++;
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
	return sweep;
}

int matrixf_decomp_hess(Matrixf* A, Matrixf* P)
{
	int i, r;
	const int n = A->size[0];
	float alpha, beta, s, t, * v, v0;

	if (A->size[1] != n || (P && (P->size[0] != n || P->size[1] != n))) {
		return -1;
	}
	if (P) {
		for (i = 0; i < n * n; i++) {
			P->data[i] = !(i % (n + 1));
		}
	}
	for (r = 0; r < n - 2; r++) {
		i = r + 2;
		while (i < n && at(A, i, r) == 0) {
			i++;
		}
		if (i < n) {
			alpha = normf(&at(A, r + 1, r), n - r - 1, 1);
			s = at(A, r + 1, r) >= 0 ? 1.0f : -1.0f;
			t = at(A, r + 1, r) + s * alpha;
			at(A, r + 1, r) = -s * alpha;
			for (i = r + 2; i < n; i++) {
				at(A, i, r) /= t;
			}
			beta = t * s / alpha;
			v0 = at(A, r + 1, r);
			v = &at(A, r + 1, r);
			v[0] = 1;
			householder_hx(A, v, beta, r + 1, n - 1, r + 1, n - 1, 1);
			householder_xh(A, v, beta, 0, n - 1, r + 1, n - 1, 1);
			if (P) {
				householder_xh(P, v, beta, 0, n - 1, r + 1, n - 1, 1);
				for (i = r + 2; i < n; i++) {
					at(A, i, r) = 0;
				}
			}
			at(A, r + 1, r) = v0;
		}
	}
	return 0;
}

int matrixf_decomp_schur_symm(Matrixf* A, Matrixf* U)
{
	const int n = A->size[0];
	int k, i, imin, imax, q, m = n - 1, sweep = 0;
	const int sweepmax = DETEGO_SCHUR_SYMM_SWEEPMAX;
	const float tol = DETEGO_SCHUR_SYMM_TOL;
	float d, f, g, x, y, cosine, sine, Xk, Xk1;

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
	while (m > 0 && sweep < sweepmax) {
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
					for (i = 0; i < n; i++) {
						Xk = at(U, i, k);
						Xk1 = at(U, i, k + 1);
						at(U, i, k) = cosine * Xk - sine * Xk1;
						at(U, i, k + 1) = sine * Xk + cosine * Xk1;
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
			sweep++;
		}
	}
	for (k = 1; k < n * n - 1; k++) {
		if (k % (n + 1)) {
			A->data[k] = 0;
		}
	}
	return sweep;
}

int matrixf_decomp_schur(Matrixf* A, Matrixf* U)
{
	const int n = A->size[0];
	int i, j, k, q, m = n - 1, sweep = 0, ad_hoc_shift;
	const int sweepmax = DETEGO_SCHUR_SWEEPMAX;
	const int ahsc = DETEGO_SCHUR_AD_HOC_SHIFT_COUNT;
	const float tol = DETEGO_SCHUR_TOL;
	const float eps = epsf(1);
	float r, s, t, x, y, z, alpha, beta, v[3] = { 1, 0, 0 };
	float sine, cosine, Xk, Xk1;

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
	while (m > 1 && sweep < sweepmax) {
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
			ad_hoc_shift = !((sweep + 1) % ahsc);
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
					alpha = hypotf(t, z);
					s = x >= 0 ? 1.0f : -1.0f;
					t = x + s * alpha;
					v[1] = y / t;
					v[2] = z / t;
					beta = t * s / alpha;
					householder_hx(A, v, beta, k + 1, k + 3, q > k ? q : k, n - 1, 1);
					householder_xh(A, v, beta, 0, (k + 4) < m ? (k + 4) : m, k + 1, k + 3, 1);
					if (U) {
						householder_xh(U, v, beta, 0, n - 1, k + 1, k + 3, 1);
					}
					x = at(A, k + 2, k + 1);
					y = at(A, k + 3, k + 1);
					if (k < m - 3) {
						z = at(A, k + 4, k + 1);
					}
				}
				alpha = hypotf(x, y);
				s = x >= 0 ? 1.0f : -1.0f;
				t = x + s * alpha;
				v[1] = y / t;
				beta = t * s / alpha;
				householder_hx(A, v, beta, m - 1, m, m - 2, n - 1, 1);
				householder_xh(A, v, beta, 0, m, m - 1, m, 1);
				if (U) {
					householder_xh(U, v, beta, 0, n - 1, m - 1, m, 1);
				}
			}
			for (k = q; k < m; k++) {
				if (fabsf(at(A, k + 1, k)) <=
					tol * (fabsf(at(A, k, k)) + fabsf(at(A, k + 1, k + 1)))) {
					at(A, k + 1, k) = 0;
				}
			}
			sweep++;
		}
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
				for (i = 0; i <= k + 1; i++) {
					Xk = at(A, i, k);
					Xk1 = at(A, i, k + 1);
					at(A, i, k) = Xk * cosine - Xk1 * sine;
					at(A, i, k + 1) = Xk * sine + Xk1 * cosine;
				}
				for (j = k; j < n; j++) {
					Xk = at(A, k, j);
					Xk1 = at(A, k + 1, j);
					at(A, k, j) = Xk * cosine - Xk1 * sine;
					at(A, k + 1, j) = Xk * sine + Xk1 * cosine;
				}
				if (U) {
					for (i = 0; i < n; i++) {
						Xk = at(U, i, k);
						Xk1 = at(U, i, k + 1);
						at(U, i, k) = Xk * cosine - Xk1 * sine;
						at(U, i, k + 1) = Xk * sine + Xk1 * cosine;
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
	return sweep;
}

int matrixf_get_eigenvectors(Matrixf* T, Matrixf* U,
	Matrixf* V, Matrixf* W, int pseudo, float* work)
{
	int i, j, k, h;
	const int n = T->size[0];
	float norm, lamre, lamim, g;
	Matrixf C = { { 0, 0 }, work }, d = { 0 };

	if (V) {
		if (V->size[0] != n || V->size[1] != n) {
			return -1;
		}
		for (i = 0; i < n * n; i++) {
			V->data[i] = !(i % (n + 1));
		}
	}
	if (W) {
		if (W->size[0] != n || W->size[1] != n) {
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
				C.size[0] = C.size[1] = k;
				matrixf_init(&d, k, 1, &at(V, 0, k), 0);
				for (j = 0; j < k; j++) {
					for (i = 0; i < k; i++) {
						at(&C, i, j) = at(T, i, j);
					}
					at(&C, j, j) -= lamre;
					at(&d, j, 0) = -at(T, j, k);
				}
				if (matrixf_solve_lu(&C, &d)) {
					return 1;
				}
				norm = normf(&at(V, 0, k), n, 1);
				for (j = 0; j < n; j++) {
					at(V, j, k) /= norm;
				}
			}
			if (W && k < n - 1) { // left eigenvector
				h = n - 1 - k;
				C.size[0] = C.size[1] = h;
				matrixf_init(&d, h, 1, &at(W, k + 1, k), 0);
				for (j = 0; j < h; j++) {
					for (i = 0; i < h; i++) {
						at(&C, j, i) = at(T, k + 1 + i, k + 1 + j);
					}
					at(&C, j, j) -= lamre;
					at(&d, j, 0) = -at(T, k, k + 1 + j);
				}
				if (matrixf_solve_lu(&C, &d)) {
					return 1;
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
						C.size[0] = C.size[1] = 2 * k;
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
							return 1;
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
						C.size[0] = C.size[1] = 2 * h;
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
							return 1;
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
						C.size[0] = C.size[1] = 2 * k;
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
							return 1;
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
						C.size[0] = C.size[1] = 2 * h;
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
							return 1;
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
	C.size[0] = C.size[1] = n;
	if (V) {
		for (i = 0; i < n * n; i++) {
			C.data[i] = V->data[i];
		}
		matrixf_multiply(U, &C, V, 1, 0, 0, 0);
	}
	if (W) {
		for (i = 0; i < n * n; i++) {
			C.data[i] = W->data[i];
		}
		matrixf_multiply(U, &C, W, 1, 0, 0, 0);
	}
	return 0;
}

int matrixf_solve_tril(Matrixf* L, Matrixf* B, Matrixf* X, int unitri)
{
	int i, j, k;
	const int m = L->size[0];
	const int n = L->size[1];
	const int h = B->size[1];
	const int q = m < n ? m : n;
	float Bik, Lii;

	if (B->size[0] != m ||
		X->size[0] != n ||
		X->size[1] != h) {
		return -1;
	}
	for (k = 0; k < h; k++) {
		for (i = 0; i < q; i++) {
			Bik = at(B, i, k);
			for (j = 0; j < i; j++) {
				Bik -= at(L, i, j) * at(B, j, k);
			}
			if (!unitri) {
				Lii = at(L, i, i);
				if (Lii == 0) {
					return 1;
				}
				at(B, i, k) = Bik / Lii;
			}
		}
	}
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
	const int m = U->size[0];
	const int n = U->size[1];
	const int h = B->size[1];
	const int q = m < n ? m : n;
	float Bik, Uii;

	if (B->size[0] != m ||
		X->size[0] != n ||
		X->size[1] != h) {
		return -1;
	}
	for (k = 0; k < h; k++) {
		for (i = q - 1; i >= 0; i--) {
			Bik = at(B, i, k);
			for (j = i + 1; j < q; j++) {
				Bik -= at(U, i, j) * at(B, j, k);
			}
			if (!unitri) {
				Uii = at(U, i, i);
				if (Uii == 0) {
					return 1;
				}
				at(B, i, k) = Bik / Uii;
			}
		}
	}
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
	const int n = A->size[0];
	const int p = B->size[1];
	float Bik;

	if (B->size[0] != n) {
		return -1;
	}
	k = matrixf_decomp_chol(A);
	if (k) {
		return k;
	}
	for (k = 0; k < p; k++) {
		for (i = 0; i < n; i++) {
			Bik = at(B, i, k);
			for (j = 0; j < i; j++) {
				Bik -= at(A, j, i) * at(B, j, k);
			}
			at(B, i, k) = Bik / at(A, i, i);
		}
		for (i = n - 1; i >= 0; i--) {
			Bik = at(B, i, k);
			for (j = i + 1; j < n; j++) {
				Bik -= at(A, i, j) * at(B, j, k);
			}
			at(B, i, k) = Bik / at(A, i, i);
		}
	}
	return 0;
}

int matrixf_solve_ltl(Matrixf* A, Matrixf* B)
{
	int i, j, k, s = 0;
	const int n = A->size[0];
	const int p = B->size[1];
	float beta, tau, t, Aii;
	Matrixf perm = { { 1, n }, 0 };

	if (B->size[0] != n || matrixf_decomp_ltl(A)) {
		return -1;
	}
	matrixf_transpose(B);
	t = A->data[0];
	A->data[0] = 0;
	perm.data = A->data;
	matrixf_permute(B, &perm, 0);
	A->data[0] = t;
	for (i = 1; i < n; i++) {
		for (j = 1; j < i; j++) {
			for (k = 0; k < p; k++) {
				at(B, k, i) -= at(A, i, j) * at(B, k, j);
			}
		}
	}
	for (j = 0; j < n - 1; j++) {
		beta = at(A, j - s, j + 1);
		s = 0;
		if (fabsf(at(A, j, j)) < fabsf(beta)) {
			s = 1;
			t = at(A, j, j + 1);
			at(A, j, j + 1) = at(A, j + 1, j + 1);
			at(A, j + 1, j + 1) = t;
			if (j + 2 < n) {
				t = at(A, j, j + 2);
				at(A, j, j + 2) = at(A, j + 1, j + 2);
				at(A, j + 1, j + 2) = t;
			}
			t = at(A, j, j);
			at(A, j, j) = beta;
			beta = t;
			swap_columns(B, j, j + 1);
		}
		if (at(A, j, j) != 0) {
			tau = beta / at(A, j, j);
			at(A, j + 1, j + 1) -= tau * at(A, j, j + 1);
			if (j + 2 < n) {
				at(A, j + 1, j + 2) -= tau * at(A, j, j + 2);
			}
			for (i = 0; i < p; i++) {
				at(B, i, j + 1) -= tau * at(B, i, j);
			}
		}
	}
	for (i = n - 1; i >= 0; i--) {
		Aii = at(A, i, i);
		if (Aii == 0) {
			return 1;
		}
		for (k = 0; k < p; k++) {
			if (i < n - 1) {
				at(B, k, i) -= at(A, i, i + 1) * at(B, k, i + 1);
			}
			if (i < n - 2) {
				at(B, k, i) -= at(A, i, i + 2) * at(B, k, i + 2);
			}
			at(B, k, i) /= Aii;
		}
	}
	for (i = n - 1; i > 0; i--) {
		for (j = i + 1; j < n; j++) {
			for (k = 0; k < p; k++) {
				at(B, k, i) -= at(A, j, i) * at(B, k, j);
			}
		}
	}
	if (n > 1) {
		A->data[n] = 0;
		for (i = 1; i < n; i++) {
			A->data[(int)A->data[i] + n] = (float)i;
		}
		perm.data = A->data + n;
		matrixf_permute(B, &perm, 0);
	}
	matrixf_transpose(B);
	return 0;
}

int matrixf_solve_lu(Matrixf* A, Matrixf* B)
{
	int i, j, k;
	const int n = A->size[0];
	const int p = B->size[1];
	float Aii, Bik;

	k = matrixf_decomp_lu(A, B);
	if (k) {
		return k;
	}
	for (k = 0; k < p; k++) {
		for (i = 1; i < n; i++) {
			Bik = at(B, i, k);
			for (j = 0; j < i; j++) {
				Bik -= at(A, i, j) * at(B, j, k);
			}
			at(B, i, k) = Bik;
		}
		for (i = n - 1; i >= 0; i--) {
			Bik = at(B, i, k);
			for (j = i + 1; j < n; j++) {
				Bik -= at(A, i, j) * at(B, j, k);
			}
			Aii = at(A, i, i);
			if (Aii == 0) {
				return 1;
			}
			at(B, i, k) = Bik / Aii;
		}
	}
	return 0;
}

int matrixf_solve_lu_banded(Matrixf* A, Matrixf* B, const int ubw)
{
	int i, j, k, p;
	const int n = A->size[0];
	const int h = B->size[1];
	float Bik, Aii;

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
			for (j = i + 1; j < p; j++) {
				Bik -= at(A, i, j) * at(B, j, k);
			}
			Aii = at(A, i, i);
			if (Aii == 0) {
				return 1;
			}
			at(B, i, k) = Bik / Aii;
		}
	}
	return 0;
}

int matrixf_solve_qr(Matrixf* A, Matrixf* B, Matrixf* X)
{
	const int m = A->size[0];
	const int n = A->size[1];
	const int p = B->size[1];

	if (B->size[0] != m ||
		X->size[0] != n ||
		X->size[1] != p) {
		return -1;
	}
	if (m < n) {
		matrixf_transpose(A);
		matrixf_decomp_qr(A, 0, 0, 0);
		matrixf_transpose(A);
		if (matrixf_solve_tril(A, B, X, 0)) {
			return 1;
		}
		matrixf_transpose(A);
		matrixf_unpack_householder_bwd(A, X, 0);
	}
	else {
		matrixf_decomp_qr(A, 0, 0, B);
		if (matrixf_solve_triu(A, B, X, 0)) {
			return 1;
		}
	}
	return 0;
}

int matrixf_solve_qrp(Matrixf* A, Matrixf* B, Matrixf* X, float tol, float* work)
{
	int i, j, k, rank;
	const int m = A->size[0];
	const int n = A->size[1];
	const int h = B->size[1];
	const int p = m < n ? m : n;
	float Bik;
	Matrixf perm = { { 1, n }, work };

	if (B->size[0] != m ||
		X->size[0] != n ||
		X->size[1] != h) {
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
	for (k = 0; k < h; k++) {
		for (i = rank - 1; i >= 0; i--) {
			Bik = at(B, i, k);
			for (j = i + 1; j < rank; j++) {
				Bik -= at(A, i, j) * at(B, j, k);
			}
			at(B, i, k) = Bik / at(A, i, i);
		}
	}
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
	matrixf_permute(X, &perm, 1);
	return 0;
}

int matrixf_solve_cod(Matrixf* A, Matrixf* B, Matrixf* X, float tol, float* work)
{
	int i, j, k, rank;
	const int m = A->size[0];
	const int n = A->size[1];
	const int h = B->size[1];
	const int p = m < n ? m : n;
	const int q = m > n ? m : n;
	float Bik;
	Matrixf perm = { { 1, n }, work };

	if (B->size[0] != m ||
		X->size[0] != n ||
		X->size[1] != h) {
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
	A->size[1] = rank;
	matrixf_decomp_qr(A, 0, 0, 0);
	for (k = 0; k < h; k++) {
		for (i = 0; i < rank; i++) {
			Bik = at(B, i, k);
			for (j = 0; j < i; j++) {
				Bik -= at(A, j, i) * at(B, j, k);
			}
			at(B, i, k) = Bik / at(A, i, i);
		}
	}
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
	matrixf_unpack_householder_bwd(A, X, 0);
	matrixf_permute(X, &perm, 1);
	A->size[0] = m;
	A->size[1] = n;
	return 0;
}

int matrixf_pseudoinv(Matrixf* A, float tol, float* work)
{
	int i, j, k, sweep;
	const int m = A->size[0];
	const int n = A->size[1];
	const int p = m < n ? m : n;
	const int q = m > n ? m : n;
	float sj, rj, Aij;
	Matrixf V = { { p, p }, work + p };
	Matrixf* X = (m < n) ? A : &V;

	if (m < n) {
		matrixf_transpose(A);
	}
	sweep = matrixf_decomp_svd_jacobi(A, 0, &V);
	if (tol < 0) {
		tol = q * epsf(normf(&at(A, 0, 0), q, 1));
	}
	for (j = 0; j < p; j++) {
		sj = normf(&at(A, 0, j), q, 1);
		if (sj > 0) {
			rj = sj > tol ? 1.0f / sj : 0;
			for (i = 0; i < q; i++) {
				at(A, i, j) /= sj;
			}
			for (i = 0; i < n; i++) {
				at(X, i, j) *= rj;
			}
		}
	}
	matrixf_transpose(A);
	for (j = 0; j < q; j++) {
		for (i = 0; i < p; i++) {
			work[i] = at(A, i, j);
		}
		for (i = 0; i < p; i++) {
			for (Aij = 0, k = 0; k < p; k++) {
				Aij += work[k] * at(&V, i, k);
			}
			at(A, i, j) = Aij;
		}
	}
	if (m < n) {
		matrixf_transpose(A);
	}
	return sweep;
}

int matrixf_exp(Matrixf* A, float* work)
{
	int i, j, k, r, z, s;
	const int n = A->size[0];
	const int q = DETEGO_EXPM_PADE_ORDER;
	float c, p, t, Xij;
	Matrixf X = { { n, n }, work + n };
	Matrixf N = { { n, n }, work + n + n * n };
	Matrixf D = { { n, n }, work + n + n * n * 2 };

	if (n != A->size[1]) {
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
		X.data[i] = N.data[i] = D.data[i] = !(i % (n + 1));
		A->data[i] /= s;
	}
	for (c = 1, k = 1; k <= q; k++) {
		c *= (float)(q - k + 1) / ((2 * q - k + 1) * k);
		for (j = 0; j < n; j++) {
			for (i = 0; i < n; i++) {
				work[i] = at(&X, i, j);
			}
			for (i = 0; i < n; i++) {
				for (Xij = 0, r = 0; r < n; r++) {
					Xij += work[r] * at(A, i, r);
				}
				at(&X, i, j) = Xij;
			}
		}
		for (i = 0; i < n * n; i++) {
			N.data[i] += c * X.data[i];
			D.data[i] += c * X.data[i] * ((k % 2) ? -1 : 1);
		}
	}
	for (i = 0; i < n * n; i++) {
		A->data[i] = N.data[i];
	}
	if (matrixf_solve_lu(&D, A)) {
		return 1;
	}
	for (k = 0; k < z; k++) {
		for (i = 0; i < n * n; i++) {
			N.data[i] = A->data[i];
		}
		matrixf_multiply(&N, &N, A, 1, 0, 0, 0);
	}
	return 0;
}

int matrixf_pow(Matrixf* A, unsigned const int p, float* work)
{
	int j, k = 0;
	const int n = A->size[0];
	unsigned int s = 1, i = 0;
	Matrixf Z = { { n, n }, work };
	Matrixf F = { { n, n }, work + n * n };

	if (A->size[1] != n) {
		return -1;
	}
	if (p == 0) {
		for (k = 0; k < n * n; k++) {
			A->data[k] = !(k % (n + 1));
		}
	}
	else if (p > 1) {
		for (j = 0; j < n * n; j++) {
			Z.data[j] = A->data[j];
		}
		while (!((p & (s << i)) >> i)) {
			matrixf_multiply(&Z, &Z, A, 1, 0, 0, 0);
			for (j = 0; j < n * n; j++) {
				Z.data[j] = A->data[j];
			}
			i++;
		}
		for (j = 0; j < n * n; j++) {
			F.data[j] = Z.data[j];
		}
		while ((s << ++i) <= p) {
			matrixf_multiply(&Z, &Z, A, 1, 0, 0, 0);
			for (j = 0; j < n * n; j++) {
				Z.data[j] = A->data[j];
			}
			if ((p & (s << i)) >> i) {
				matrixf_multiply(&F, &Z, A, 1, 0, 0, 0);
				for (j = 0; j < n * n; j++) {
					F.data[j] = A->data[j];
				}
			}
		}
	}
	return 0;
}

int matrixf_multiply(Matrixf* A, Matrixf* B, Matrixf* C,
	const float alpha, const float beta, const int transA, const int transB)
{
	int i, j, k;
	const int m = C->size[0];
	const int n = C->size[1];
	const int p = (transA) ? A->size[0] : A->size[1];
	float b, c;

	for (i = 0; i < m * n; i++) {
		C->data[i] *= beta;
	}
	if (!alpha) {
		return 0;
	}
	if (transA) {
		if (transB) { // C = alpha * A' * B' + beta * C
			if (m != A->size[1] || p != B->size[1] || n != B->size[0]) {
				return -1;
			}
			for (j = 0; j < n; j++) {
				for (i = 0; i < m; i++) {
					c = at(C, i, j);
					for (k = 0; k < p; k++) {
						c += alpha * at(A, k, i) * at(B, j, k);
					}
					at(C, i, j) = c;
				}
			}
		}
		else { // C = alpha * A' * B + beta * C
			if (m != A->size[1] || p != B->size[0] || n != B->size[1]) {
				return -1;
			}
			for (j = 0; j < n; j++) {
				for (i = 0; i < m; i++) {
					c = at(C, i, j);
					for (k = 0; k < p; k++) {
						c += alpha * at(A, k, i) * at(B, k, j);
					}
					at(C, i, j) = c;
				}
			}
		}
	}
	else {
		if (transB) { // C = alpha * A * B' + beta * C
			if (m != A->size[0] || p != B->size[1] || n != B->size[0]) {
				return -1;
			}
			for (k = 0; k < p; k++) {
				for (j = 0; j < n; j++) {
					b = at(B, j, k);
					for (i = 0; i < m; i++) {
						at(C, i, j) += alpha * at(A, i, k) * b;
					}
				}
			}
		}
		else { // C = alpha * A * B + beta * C
			if (m != A->size[0] || p != B->size[0] || n != B->size[1]) {
				return -1;
			}
			for (j = 0; j < n; j++) {
				for (k = 0; k < p; k++) {
					b = at(B, k, j);
					for (i = 0; i < m; i++) {
						at(C, i, j) += alpha * at(A, i, k) * b;
					}
				}
			}
		}
	}
	return 0;
}