#include "detego.h"

// This function swaps the columns i and j of the matrix A.
static void swap_columns(Matrixf* A, int i, int j)
{
	int k;
	const int m = A->size[0];
	float* x = &_(A, 0, i);
	float* y = &_(A, 0, j);
	float xk;

	for (k = 0; k < m; k++) {
		xk = x[k];
		x[k] = y[k];
		y[k] = xk;
	}
}

// Householder transformation X(i0:iend,j0:jend) = H*X(i0:iend,j0:jend),
// where H = I - beta*v*v'; stride is the increment of v.
static void apply_householder_left(Matrixf* X, float* v, const float beta,
	const int i0, const int iend, const int j0, const int jend, const int stride)
{
	int i, j;
	float h;

	for (j = j0; j <= jend; j++) {
		h = _(X, i0, j) * v[0];
		for (i = i0 + 1; i <= iend; i++) {
			h += _(X, i, j) * v[(i - i0) * stride];
		}
		_(X, i0, j) -= beta * h;
		for (i = i0 + 1; i <= iend; i++) {
			_(X, i, j) -= beta * h * v[(i - i0) * stride];
		}
	}
}

// Householder transformation X(i0:iend,j0:jend) = X(i0:iend,j0:jend)*H,
// where H = I - beta*v*v'; stride is the increment of v.
static void apply_householder_right(Matrixf* X, float* v, const float beta,
	const int i0, const int iend, const int j0, const int jend, const int stride)
{
	int i, j;
	float h;

	for (i = i0; i <= iend; i++) {
		h = _(X, i, j0) * v[0];
		for (j = j0 + 1; j <= jend; j++) {
			h += _(X, i, j) * v[(j - j0) * stride];
		}
		_(X, i, j0) -= beta * h;
		for (j = j0 + 1; j <= jend; j++) {
			_(X, i, j) -= beta * h * v[(j - j0) * stride];
		}
	}
}

void matrixf_init(Matrixf* A, int rows, int cols, float* data, int ordmem)
{
	A->data = data;
	A->size[ordmem != 0] = rows;
	A->size[ordmem == 0] = cols;
	if (ordmem) {
		matrixf_transpose(A);
	}
}

int matrixf_permute(Matrixf* A, Matrixf* p, const int transP)
{
	int i, j, k, t;
	const int m = transP ? p->size[1] : p->size[0];
	const int n = transP ? p->size[0] : p->size[1];
	const int h = m > n ? m : n;
	float* x = p->data;

	if ((A->size[0] != m || n != 1) &&
		(A->size[1] != n || m != 1)) {
		return -1;
	}
	if (transP && h > 1) {
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
			for (v = _(A, j, i), k = 0; k < j; k++) {
				v -= _(A, k, i) * _(A, k, j);
			}
			if (i == j) {
				if (v > 0) {
					r = sqrtf(v);
				}
				else {
					return 1;
				}
			}
			_(A, j, i) = v / r;
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
		_(A, 0, j) = (float)j;
	}
	for (j = 0; j < n; j++) {
		if (j) {
			h[0] = (j == 1) ? _(A, 0, 1) : _(A, 0, 1) * _(A, j, 1);
			for (k = 1; k < j; k++) {
				Ljj = (j == k + 1) ? 1.0f : _(A, j, k + 1);
				h[k] = _(A, k - 1, k) * _(A, j, k - 1) * (k > 1)
					+ _(A, k, k) * _(A, j, k) + _(A, k, k + 1) * Ljj;
			}
			for (s = 0, k = 1; k < j; k++) {
				s += _(A, j, k) * h[k];
			}
			h[j] = _(A, j, j) - s;
			_(A, j, j) = h[j] - _(A, j - 1, j) * _(A, j, j - 1) * (j > 1);
			for (i = j + 1; i < n; i++) {
				for (s = 0, k = 1; k <= j; k++) {
					s += _(A, i, k) * h[k];
				}
				h[i] = _(A, j, i) - s;
			}
		}
		if (j < n - 1) {
			for (m = j + 1, k = j + 1; k < n; k++) {
				if (fabsf(h[k]) > fabsf(h[m])) {
					m = k;
				}
			}
			for (k = 0; k < n; k++) {
				s = _(A, m, k);
				_(A, m, k) = _(A, j + 1, k);
				_(A, j + 1, k) = s;
			}
			swap_columns(A, m, j + 1);
			if (!j) {
				P1 = _(A, 0, 1);
			}
			_(A, j, j + 1) = h[j + 1];
			if (j < n - 2) {
				for (i = j + 2; i < n; i++) {
					_(A, i, j + 1) = h[m] ? h[i] / h[j + 1] : 0;
				}
			}
		}
	}
	A->data[0] = A0;
	if (n > 1) {
		A->data[1] = P1;
	}
	for (j = 2; j < n; j++) {
		_(A, j, 0) = _(A, 0, j);
		for (i = 0; i < j - 1; i++) {
			_(A, i, j) = 0;
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
		a = fabsf(_(A, i, j));
		for (j = i + 1; j < n; j++) {
			b = fabsf(_(A, i, j));
			if (b > a) {
				k = j;
				a = b;
			}
		}
		swap_columns(A, k, i);
		swap_columns(P, k, i);
		for (k = i + 1; k < n; k++) {
			a = _(A, i, i);
			if (a != 0) {
				_(A, i, k) /= a;
			}
			for (j = i + 1; j < n; j++) {
				_(A, j, k) -= _(A, j, i) * _(A, i, k);
			}
		}
	}
	matrixf_transpose(A);
	matrixf_transpose(P);
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
				c += _(A, i, k) * _(A, i, k);
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
		while (i < m && _(A, i, r) == 0) {
			i++;
		}
		if (i < m) {
			alpha = get_norm2(&_(A, r, r), m - r, 1);
			s = _(A, r, r) >= 0 ? 1.0f : -1.0f;
			t = _(A, r, r) + s * alpha;
			_(A, r, r) = -s * alpha;
			for (i = r + 1; i < m; i++) {
				_(A, i, r) /= t;
			}
			beta = t * s / alpha;
			v0 = _(A, r, r);
			v = &_(A, r, r);
			v[0] = 1;
			apply_householder_left(A, v, beta, r, m - 1, r + 1, n - 1, 1);
			if (B) {
				apply_householder_left(B, v, beta, r, m - 1, 0, B->size[1] - 1, 1);
			}
			if (Q && q == m) {
				apply_householder_right(Q, v, beta, 0, m - 1, r, m - 1, 1);
				for (i = r + 1; i < m; i++) {
					_(A, i, r) = 0;
				}
			}
			_(A, r, r) = v0;
		}
		if (P && r + 1 <= rmax) {
			for (cm = 0, km = 0, k = r + 1; k < n; k++) {
				for (c = 0, i = r + 1; i < m; i++) {
					c += _(A, i, k) * _(A, i, k);
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
		matrixf_accumulate_bwd(A, Q);
		for (k = 0; k < n; k++) {
			for (i = 0; i < n; i++) {
				_(&A_econ, i, k) = (i <= k) ? _(A, i, k) : 0.0f;
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
			_(P, i, 0) = 0;
			_(P, k, i) = 1;
		}
	}
	return 0;
}

int matrixf_accumulate_fwd(Matrixf* H, Matrixf* B)
{
	int i, k;
	const int m = H->size[0];
	const int n = H->size[1];
	const int p = B->size[1];
	const int kmax = m - 1 < n ? m - 2 : n - 1;
	float alpha, * v, v0;

	if (B->size[0] != m) {
		return -1;
	}
	for (k = 0; k <= kmax; k++) {
		for (alpha = 1, i = k + 1; i < m; i++) {
			alpha += _(H, i, k) * _(H, i, k);
		}
		if (alpha > 1) {
			v0 = _(H, k, k);
			v = &_(H, k, k);
			v[0] = 1;
			apply_householder_left(B, v, 2.0f / alpha, k, m - 1, 0, p - 1, 1);
			_(H, k, k) = v0;
		}
	}
	return 0;
}

int matrixf_accumulate_bwd(Matrixf* H, Matrixf* B)
{
	int i, k;
	const int m = H->size[0];
	const int n = H->size[1];
	const int p = B->size[1];
	const int kmax = m - 1 < n ? m - 2 : n - 1;
	float alpha, * v, v0;

	if (B->size[0] != m) {
		return -1;
	}
	for (k = kmax; k >= 0; k--) {
		for (alpha = 1, i = k + 1; i < m; i++) {
			alpha += _(H, i, k) * _(H, i, k);
		}
		if (alpha > 1) {
			v0 = _(H, k, k);
			v = &_(H, k, k);
			v[0] = 1;
			apply_householder_left(B, v, 2.0f / alpha, k, m - 1, 0, p - 1, 1);
			_(H, k, k) = v0;
		}
	}
	return 0;
}

int matrixf_bidiagonalize(Matrixf* A, Matrixf* U, Matrixf* V)
{
	int i, k, q = A->size[0];
	const int m = A->size[0];
	const int n = A->size[1];
	const int kmax = m - 1 < n ? m - 2 : n - 1;
	float alpha, beta, s, t, * v, v0;
	Matrixf A_econ = { { n, n }, A->data };

	if (m < n) {
		matrixf_transpose(A);
		k = matrixf_bidiagonalize(A, V, U);
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
		while (i < m && _(A, i, k) == 0) {
			i++;
		}
		if (i < m) {
			alpha = get_norm2(&_(A, k, k), m - k, 1);
			s = _(A, k, k) >= 0 ? 1.0f : -1.0f;
			t = _(A, k, k) + s * alpha;
			_(A, k, k) = -s * alpha;
			for (i = k + 1; i < m; i++) {
				_(A, i, k) /= t;
			}
			beta = t * s / alpha;
			v0 = _(A, k, k);
			v = &_(A, k, k);
			v[0] = 1;
			apply_householder_left(A, v, beta, k, m - 1, k + 1, n - 1, 1);
			if (U && q == m) {
				apply_householder_right(U, v, beta, 0, m - 1, k, m - 1, 1);
			}
			if (q == m) {
				for (i = k + 1; i < m; i++) {
					_(A, i, k) = 0;
				}
			}
			_(A, k, k) = v0;
		}
		i = k + 2;
		while (i < n && _(A, k, i) == 0) {
			i++;
		}
		if (i < n && k + 2 < n) {
			alpha = get_norm2(&_(A, k, k + 1), n - k - 1, m);
			s = _(A, k, k + 1) >= 0 ? 1.0f : -1.0f;
			t = _(A, k, k + 1) + s * alpha;
			_(A, k, k + 1) = -s * alpha;
			for (i = k + 2; i < n; i++) {
				_(A, k, i) /= t;
			}
			beta = t * s / alpha;
			v0 = _(A, k, k + 1);
			v = &_(A, k, k + 1);
			v[0] = 1;
			apply_householder_right(A, v, beta, k + 1, m - 1, k + 1, n - 1, m);
			if (V) {
				apply_householder_right(V, v, beta, 0, n - 1, k + 1, n - 1, m);
			}
			for (i = k + 2; i < n; i++) {
				_(A, k, i) = 0;
			}
			_(A, k, k + 1) = v0;
		}
	}
	if (U && q < m) {
		matrixf_accumulate_bwd(A, U);
		for (k = 0; k < n; k++) {
			for (i = 0; i < n; i++) {
				_(&A_econ, i, k) = (i <= k) ? _(A, i, k) : 0.0f;
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
	if (matrixf_bidiagonalize(A, U, V)) {
		return -1;
	}
	for (norm1 = fabsf(_(A, 0, 0)), j = 1; j < n; j++) {
		tmp = fabsf(_(A, j - 1, j)) + fabsf(_(A, j, j));
		if (norm1 < tmp) {
			norm1 = tmp;
		}
	}
	small = tol * norm1;
	while (r > 0 && sweep < sweepmax) {
		while (r > 0 && fabsf(_(A, r - 1, r)) <= small) {
			r--;
		}
		if (r > 0) {
			q = i = r;
			while (q > 0 && fabsf(_(A, q - 1, q)) > small) {
				q--;
			}
			while (i >= q && fabsf(_(A, i, i)) > small) {
				i--;
			}
			if (i == q - 1) {
				c00 = c01 = c11 = 0;
				for (k = (r - 2 < 0 ? 0 : r - 2); k <= r; k++) {
					Xi = _(A, k, r - 1);
					Xj = _(A, k, r);
					c00 += Xi * Xi;
					c01 += Xi * Xj;
					c11 += Xj * Xj;
				}
				y = 0.5f * (c00 - c11);
				z = sqrtf(y * y + c01 * c01);
				mu = fabsf(y + z) < fabsf(y - z) ? c11 + y + z : c11 + y - z;
				y = _(A, q, q) * _(A, q, q) - mu;
				z = _(A, q, q + 1) * _(A, q, q);
				for (j = q; j < r; j++) {
					get_givensrot(y, z, &cosine, &sine);
					for (k = (j - 1 < 0 ? 0 : j - 1); k <= j + 1; k++) {
						Xi = _(A, k, j);
						Xj = _(A, k, j + 1);
						_(A, k, j) = cosine * Xi - sine * Xj;
						_(A, k, j + 1) = sine * Xi + cosine * Xj;
					}
					if (V) {
						for (k = 0; k < n; k++) {
							Xi = _(V, k, j);
							Xj = _(V, k, j + 1);
							_(V, k, j) = cosine * Xi - sine * Xj;
							_(V, k, j + 1) = sine * Xi + cosine * Xj;
						}
					}
					y = _(A, j, j);
					z = _(A, j + 1, j);
					get_givensrot(y, z, &cosine, &sine);
					for (k = j; k <= (j + 2 < r ? j + 2 : r); k++) {
						Xi = _(A, j, k);
						Xj = _(A, j + 1, k);
						_(A, j, k) = cosine * Xi - sine * Xj;
						_(A, j + 1, k) = sine * Xi + cosine * Xj;
					}
					if (U) {
						for (k = 0; k < m; k++) {
							Xi = _(U, k, j);
							Xj = _(U, k, j + 1);
							_(U, k, j) = cosine * Xi - sine * Xj;
							_(U, k, j + 1) = sine * Xi + cosine * Xj;
						}
					}
					if (j < r - 1) {
						y = _(A, j, j + 1);
						z = _(A, j, j + 2);
					}
				}
				sweep++;
			}
			else if (i < r) {
				for (j = i + 1; j <= r; j++) {
					get_givensrot(-_(A, j, j), _(A, i, j), &cosine, &sine);
					for (k = j; k <= (j + 1 < r ? j + 1 : r); k++) {
						Xi = _(A, i, k);
						Xj = _(A, j, k);
						_(A, i, k) = cosine * Xi - sine * Xj;
						_(A, j, k) = sine * Xi + cosine * Xj;
					}
					if (U) {
						for (k = 0; k < m; k++) {
							Xi = _(U, k, i);
							Xj = _(U, k, j);
							_(U, k, i) = cosine * Xi - sine * Xj;
							_(U, k, j) = sine * Xi + cosine * Xj;
						}
					}
				}
			}
			else {
				for (j = r - 1; j >= q; j--) {
					get_givensrot(_(A, j, j), _(A, j, r), &cosine, &sine);
					for (k = (j - 1 > q ? j - 1 : q); k <= j; k++) {
						Xi = _(A, k, j);
						Xj = _(A, k, r);
						_(A, k, j) = cosine * Xi - sine * Xj;
						_(A, k, r) = sine * Xi + cosine * Xj;
					}
					if (V) {
						for (k = 0; k < n; k++) {
							Xi = _(V, k, j);
							Xj = _(V, k, r);
							_(V, k, j) = cosine * Xi - sine * Xj;
							_(V, k, r) = sine * Xi + cosine * Xj;
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
		s[j] = _(A, j, j);
		if (j > 1) {
			for (i = 0; i < n; i++) {
				_(A, i, j) = 0;
			}
		}
		if (s[j] < 0) {
			s[j] *= -1.0f;
			if (U) {
				for (i = 0; i < m; i++) {
					_(U, i, j) *= -1.0f;
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
			_(A, j, j) = s[j];
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
					a += _(A, i, j) * _(A, i, j);
					b += _(A, i, k) * _(A, i, k);
					p += _(A, i, j) * _(A, i, k);
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
						Xij = _(A, i, j);
						Xik = _(A, i, k);
						_(A, i, j) = +Xij * cosine + Xik * sine;
						_(A, i, k) = -Xij * sine + Xik * cosine;
					}
					if (V) {
						for (i = 0; i < n; i++) {
							Xij = _(V, i, j);
							Xik = _(V, i, k);
							_(V, i, j) = +Xij * cosine + Xik * sine;
							_(V, i, k) = -Xij * sine + Xik * cosine;
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
			if (_(A, j, j) < 0) {
				_(A, j, j) *= -1.0f;
				for (i = 0; i < m; i++) {
					_(U, i, j) *= -1.0f;
				}
			}
			for (i = 0; i < j; i++) {
				_(A, i, j) = 0;
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
		while (i < n && _(A, i, r) == 0) {
			i++;
		}
		if (i < n) {
			alpha = get_norm2(&_(A, r + 1, r), n - r - 1, 1);
			s = _(A, r + 1, r) >= 0 ? 1.0f : -1.0f;
			t = _(A, r + 1, r) + s * alpha;
			_(A, r + 1, r) = -s * alpha;
			for (i = r + 2; i < n; i++) {
				_(A, i, r) /= t;
			}
			beta = t * s / alpha;
			v0 = _(A, r + 1, r);
			v = &_(A, r + 1, r);
			v[0] = 1;
			apply_householder_left(A, v, beta, r + 1, n - 1, r + 1, n - 1, 1);
			apply_householder_right(A, v, beta, 0, n - 1, r + 1, n - 1, 1);
			if (P) {
				apply_householder_right(P, v, beta, 0, n - 1, r + 1, n - 1, 1);
			}
			_(A, r + 1, r) = v0;
			for (i = r + 2; i < n; i++) {
				_(A, i, r) = 0;
			}
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
	while (m > 0 && sweep < sweepmax) {
		while (m > 0 && _(A, m, m - 1) == 0) {
			m--;
		}
		if (m > 0) {
			q = m - 1;
			while (q > 0 && _(A, q, q - 1) != 0) {
				q--;
			}
			d = 0.5f * (_(A, m - 1, m - 1) - _(A, m, m));
			f = d < 0 ? -1.0f : 1.0f;
			g = _(A, m, m - 1) * _(A, m, m - 1);
			x = _(A, q, q) - _(A, m, m) + g / (d + f * sqrtf(d * d + g));
			y = _(A, q + 1, q);
			for (k = q; k < m; k++) {
				get_givensrot(x, y, &cosine, &sine);
				imin = k - 1 < 0 ? 0 : k - 1;
				imax = k + 2 > n - 1 ? n - 1 : k + 2;
				for (i = imin; i <= imax; i++) {
					Xk = _(A, k, i);
					Xk1 = _(A, k + 1, i);
					_(A, k, i) = cosine * Xk - sine * Xk1;
					_(A, k + 1, i) = sine * Xk + cosine * Xk1;
				}
				for (i = imin; i <= imax; i++) {
					Xk = _(A, i, k);
					Xk1 = _(A, i, k + 1);
					_(A, i, k) = cosine * Xk - sine * Xk1;
					_(A, i, k + 1) = sine * Xk + cosine * Xk1;
				}
				if (U) {
					for (i = 0; i < n; i++) {
						Xk = _(U, i, k);
						Xk1 = _(U, i, k + 1);
						_(U, i, k) = cosine * Xk - sine * Xk1;
						_(U, i, k + 1) = sine * Xk + cosine * Xk1;
					}
				}
				if (k < m - 1) {
					x = _(A, k + 1, k);
					y = _(A, k + 2, k);
				}
			}
			for (k = q; k < m; k++) {
				if (fabsf(_(A, k + 1, k)) <=
					tol * (fabsf(_(A, k, k)) + fabsf(_(A, k + 1, k + 1)))) {
					_(A, k + 1, k) = 0;
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
	const float eps = EPS(1);
	float r, s, t, x, y, z, alpha, beta, v[3] = { 1, 0, 0 };
	float sine, cosine, Xk, Xk1;

	if (matrixf_decomp_hess(A, U)) {
		return -1;
	}
	while (m > 1 && sweep < sweepmax) {
		while (m > 1 && (_(A, m, m - 1) == 0 || _(A, m - 1, m - 2) == 0)) {
			if (_(A, m, m - 1) == 0) {
				m--;
			}
			else if (_(A, m - 1, m - 2) == 0) {
				m -= 2;
			}
		}
		if (m > 0) {
			q = m - 1;
			while (q > 0 && _(A, q, q - 1) != 0) {
				q--;
			}
			ad_hoc_shift = !((sweep + 1) % ahsc);
			if (m - q > 1) {
				if (ad_hoc_shift) {
					x = 0.5f * (_(A, m - 1, m - 1) - _(A, m, m));
					y = x * x + _(A, m - 1, m) * _(A, m, m - 1);
					s = t = x = _(A, m, m) + x;
					if (y > 0) {
						z = sqrtf(y);
						s -= z;
						t += z;
						x = fabsf(s - _(A, m, m)) < fabsf(t - _(A, m, m)) ? s : t;
					}
					if (x == 0) {
						x = eps;
					}
					s = x + x;
					t = x * x;
				}
				else {
					s = _(A, m - 1, m - 1) + _(A, m, m);
					t = _(A, m - 1, m - 1) * _(A, m, m) - _(A, m - 1, m) * _(A, m, m - 1);
				}
				x = _(A, q, q) * _(A, q, q) + _(A, q, q + 1) * _(A, q + 1, q) - s * _(A, q, q) + t;
				y = _(A, q + 1, q) * (_(A, q, q) + _(A, q + 1, q + 1) - s);
				z = _(A, q + 1, q) * _(A, q + 2, q + 1);
				for (k = q - 1; k <= m - 3; k++) {
					t = hypotf(x, y);
					alpha = hypotf(t, z);
					s = x >= 0 ? 1.0f : -1.0f;
					t = x + s * alpha;
					v[1] = y / t;
					v[2] = z / t;
					beta = t * s / alpha;
					apply_householder_left(A, v, beta, k + 1, k + 3, q > k ? q : k, n - 1, 1);
					apply_householder_right(A, v, beta, 0, (k + 4) < m ? (k + 4) : m, k + 1, k + 3, 1);
					if (U) {
						apply_householder_right(U, v, beta, 0, n - 1, k + 1, k + 3, 1);
					}
					x = _(A, k + 2, k + 1);
					y = _(A, k + 3, k + 1);
					if (k < m - 3) {
						z = _(A, k + 4, k + 1);
					}
				}
				alpha = hypotf(x, y);
				s = x >= 0 ? 1.0f : -1.0f;
				t = x + s * alpha;
				v[1] = y / t;
				beta = t * s / alpha;
				apply_householder_left(A, v, beta, m - 1, m, m - 2, n - 1, 1);
				apply_householder_right(A, v, beta, 0, m, m - 1, m, 1);
				if (U) {
					apply_householder_right(U, v, beta, 0, n - 1, m - 1, m, 1);
				}
			}
			for (k = q; k < m; k++) {
				if (fabsf(_(A, k + 1, k)) <=
					tol * (fabsf(_(A, k, k)) + fabsf(_(A, k + 1, k + 1)))) {
					_(A, k + 1, k) = 0;
				}
			}
			sweep++;
		}
	}
	// Trangularize all 2-by-2 diagonal blocks in A that have real eigenvalues,
	// and transform the blocks with complex eigenvalues so that the real part
	// of the eigenvalues appears on the main diagonal. 
	for (k = 0; k < n - 1; k++) {
		if (_(A, k + 1, k) != 0) {
			sine = 0;
			x = 0.5f * (_(A, k, k) - _(A, k + 1, k + 1));
			y = x * x + _(A, k + 1, k) * _(A, k, k + 1);
			if (y >= 0) { // real eigenvalues
				r = _(A, k, k + 1);
				s = _(A, k, k) - _(A, k + 1, k + 1) - x + sqrtf(y);
				t = hypotf(r, s);
				cosine = r / t;
				sine = s / t;
			}
			else if (_(A, k, k) != _(A, k + 1, k + 1)) {
				r = (_(A, k + 1, k) + _(A, k, k + 1)) /
					(_(A, k + 1, k + 1) - _(A, k, k));
				t = r - sqrtf(r * r + 1);
				cosine = 1.0f / sqrtf(t * t + 1);
				sine = cosine * t;
			}
			if (sine != 0) {
				for (i = 0; i <= k + 1; i++) {
					Xk = _(A, i, k);
					Xk1 = _(A, i, k + 1);
					_(A, i, k) = Xk * cosine - Xk1 * sine;
					_(A, i, k + 1) = Xk * sine + Xk1 * cosine;
				}
				for (j = k; j < n; j++) {
					Xk = _(A, k, j);
					Xk1 = _(A, k + 1, j);
					_(A, k, j) = Xk * cosine - Xk1 * sine;
					_(A, k + 1, j) = Xk * sine + Xk1 * cosine;
				}
				if (U) {
					for (i = 0; i < n; i++) {
						Xk = _(U, i, k);
						Xk1 = _(U, i, k + 1);
						_(U, i, k) = Xk * cosine - Xk1 * sine;
						_(U, i, k + 1) = Xk * sine + Xk1 * cosine;
					}
				}
			}
			if (y >= 0) {
				_(A, k + 1, k) = 0;
			}
		}
		for (i = k + 2; i < n; i++) {
			_(A, i, k) = 0;
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
		if (k == n - 1 || _(T, k + 1, k) == 0) { // real eigenvalue
			lamre = _(T, k, k) + EPS(_(T, k, k));
			if (V && k > 0) { // right eigenvector
				C.size[0] = C.size[1] = k;
				matrixf_init(&d, k, 1, &_(V, 0, k), 0);
				for (j = 0; j < k; j++) {
					for (i = 0; i < k; i++) {
						_(&C, i, j) = _(T, i, j);
					}
					_(&C, j, j) -= lamre;
					_(&d, j, 0) = -_(T, j, k);
				}
				if (matrixf_solve_lu(&C, &d)) {
					return 1;
				}
				norm = get_norm2(&_(V, 0, k), n, 1);
				for (j = 0; j < n; j++) {
					_(V, j, k) /= norm;
				}
			}
			if (W && k < n - 1) { // left eigenvector
				h = n - 1 - k;
				C.size[0] = C.size[1] = h;
				matrixf_init(&d, h, 1, &_(W, k + 1, k), 0);
				for (j = 0; j < h; j++) {
					for (i = 0; i < h; i++) {
						_(&C, j, i) = _(T, k + 1 + i, k + 1 + j);
					}
					_(&C, j, j) -= lamre;
					_(&d, j, 0) = -_(T, k, k + 1 + j);
				}
				if (matrixf_solve_lu(&C, &d)) {
					return 1;
				}
				norm = get_norm2(&_(W, 0, k), n, 1);
				for (j = 0; j < n; j++) {
					_(W, j, k) /= norm;
				}
			}
			k += 1;
		}
		else { // complex conjugate pair of eigenvalues
			// It is assumed that the 2-by-2 blocks have been transformed so 
			// that the real part of the eigenvalues appears on the diagonal
			lamre = _(T, k, k);
			lamim = sqrtf(-_(T, k + 1, k) * _(T, k, k + 1));
			if (pseudo) { // pseudo-eigenvectors
				g = _(T, k + 1, k) / lamim;
				if (V) { // right eigenvectors
					if (k > 0) {
						C.size[0] = C.size[1] = 2 * k;
						matrixf_init(&d, 2 * k, 1, &_(V, 0, k), 0);
						for (j = 0; j < k; j++) {
							for (i = 0; i < k; i++) {
								_(&C, i, j) = _(&C, k + i, k + j) = _(T, i, j);
								_(&C, i, k + j) = _(&C, k + i, j) = 0;
							}
							_(&C, j, j) -= _(T, k, k);
							_(&C, k + j, k + j) -= _(T, k + 1, k + 1);
							_(&C, j, k + j) = +lamim;
							_(&C, k + j, j) = -lamim;
							_(&d, j, 0) = -_(T, j, k) - _(T, k + 1, k) * _(T, j, k + 1);
							_(&d, k + j, 0) = -lamim * _(T, j, k) + g * _(T, j, k + 1);
						}
						if (matrixf_solve_lu(&C, &d)) {
							return 1;
						}
						for (j = k - 1; j >= 0; j--) {
							_(V, j, k + 1) = d.data[k + j];
							d.data[k + j] = 0;
						}
					}
					_(V, k, k) = 1;
					_(V, k + 1, k) = _(T, k + 1, k);
					_(V, k, k + 1) = lamim;
					_(V, k + 1, k + 1) = -g;
					norm = get_norm2(&_(V, 0, k), 2 * n, 1);
					for (j = 0; j < n; j++) {
						_(V, j, k) /= norm;
						_(V, j, k + 1) /= norm;
					}
				}
				if (W) { // left eigenvectors
					if (k < n - 2) {
						h = n - 2 - k;
						C.size[0] = C.size[1] = 2 * h;
						matrixf_init(&d, 2 * h, 1, &_(W, k + 2, k), 0);
						for (j = 0; j < h; j++) {
							for (i = 0; i < h; i++) {
								_(&C, j, i) = _(&C, h + j, h + i) =
									_(T, k + 2 + i, k + 2 + j);
								_(&C, i, h + j) = _(&C, h + i, j) = 0;
							}
							_(&C, j, j) -= _(T, k, k);
							_(&C, h + j, h + j) -= _(T, k + 1, k + 1);
							_(&C, h + j, j) = +lamim;
							_(&C, j, h + j) = -lamim;
							_(&d, j, 0) = g * _(T, k, k + 2 + j) - lamim * _(T, k + 1, k + 2 + j);
							_(&d, h + j, 0) = -_(T, k + 1, k) * _(T, k, k + 2 + j) - _(T, k + 1, k + 2 + j);
						}
						if (matrixf_solve_lu(&C, &d)) {
							return 1;
						}
						for (j = h - 1; j >= 0; j--) {
							_(W, k + 2 + j, k + 1) = _(W, j, k + 1);
							_(W, j, k + 1) = 0;
						}
					}
					_(W, k, k) = -g;
					_(W, k + 1, k) = lamim;
					_(W, k, k + 1) = _(T, k + 1, k);
					_(W, k + 1, k + 1) = 1;
					norm = get_norm2(&_(W, 0, k), 2 * n, 1);
					for (j = 0; j < n; j++) {
						_(W, j, k) /= norm;
						_(W, j, k + 1) /= norm;
					}
				}
			}
			else { // complex eigenvectors
				if (V) { // right eigenvectors
					if (k > 0) {
						C.size[0] = C.size[1] = 2 * k;
						matrixf_init(&d, 2 * k, 1, &_(V, 0, k), 0);
						for (j = 0; j < k; j++) {
							for (i = 0; i < k; i++) {
								_(&C, i, j) = _(&C, k + i, k + j) = _(T, i, j);
								_(&C, i, k + j) = _(&C, k + i, j) = 0;
							}
							_(&C, j, j) -= lamre;
							_(&C, k + j, k + j) -= lamre;
							_(&C, k + j, j) = -lamim;
							_(&C, j, k + j) = +lamim;
							_(&d, j, 0) = -_(T, j, k + 1) * _(T, k + 1, k);
							_(&d, k + j, 0) = -_(T, j, k) * lamim;
						}
						if (matrixf_solve_lu(&C, &d)) {
							return 1;
						}
						for (j = k - 1; j >= 0; j--) {
							_(V, j, k + 1) = d.data[k + j];
							d.data[k + j] = 0;
						}
					}
					_(V, k, k) = _(V, k + 1, k + 1) = 0;
					_(V, k + 1, k) = _(T, k + 1, k);
					_(V, k, k + 1) = lamim;
					norm = get_norm2(&_(V, 0, k), 2 * n, 1);
					for (j = 0; j < 2 * n; j++) {
						_(V, j, k) /= norm;
					}
				}
				if (W) { // left eigenvectors
					if (k < n - 2) {
						h = n - 2 - k;
						C.size[0] = C.size[1] = 2 * h;
						matrixf_init(&d, 2 * h, 1, &_(W, k + 2, k), 0);
						for (j = 0; j < h; j++) {
							for (i = 0; i < h; i++) {
								_(&C, j, i) = _(&C, h + j, h + i) =
									_(T, k + 2 + i, k + 2 + j);
								_(&C, i, h + j) = _(&C, h + i, j) = 0;
							}
							_(&C, j, j) -= lamre;
							_(&C, h + j, h + j) -= lamre;
							_(&C, h + j, j) = +lamim;
							_(&C, j, h + j) = -lamim;
							_(&d, j, 0) = -_(T, k + 1, k + 2 + j) * _(T, k, k + 1);
							_(&d, h + j, 0) = _(T, k, k + 2 + j) * lamim;
						}
						if (matrixf_solve_lu(&C, &d)) {
							return 1;
						}
						for (j = h - 1; j >= 0; j--) {
							_(W, k + 2 + j, k + 1) = _(W, j, k + 1);
							_(W, j, k + 1) = 0;
						}
					}
					_(W, k, k) = _(W, k + 1, k + 1) = 0;
					_(W, k + 1, k) = _(T, k, k + 1);
					_(W, k, k + 1) = -lamim;
					norm = get_norm2(&_(W, 0, k), 2 * n, 1);
					for (j = 0; j < 2 * n; j++) {
						_(W, j, k) /= norm;
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
			Bik = _(B, i, k);
			for (j = 0; j < i; j++) {
				Bik -= _(L, i, j) * _(X, j, k);
			}
			if (!unitri) {
				Lii = _(L, i, i);
				if (Lii == 0) {
					return 1;
				}
				Bik /= Lii;
			}
			_(X, i, k) = Bik;
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
			Bik = _(B, i, k);
			for (j = i + 1; j < q; j++) {
				Bik -= _(U, i, j) * _(X, j, k);
			}
			if (!unitri) {
				Uii = _(U, i, i);
				if (Uii == 0) {
					return 1;
				}
				Bik /= Uii;
			}
			_(X, i, k) = Bik;
		}
	}
	return 0;
}

int matrixf_solve_tridiag(Matrixf* T, Matrixf* B)
{
	int i, j, k;
	const int n = T->size[0];
	const int p = B->size[1];
	float tau, t;

	if (T->size[1] != n || B->size[0] != n) {
		return -1;
	}
	matrixf_transpose(B);
	for (j = 0; j < n - 1; j++) {
		if (j + 2 < n) {
			_(T, j, j + 2) = 0;
		}
		if (fabsf(_(T, j, j)) < fabsf(_(T, j + 1, j))) {
			t = _(T, j, j);
			_(T, j, j) = _(T, j + 1, j);
			_(T, j + 1, j) = t;
			t = _(T, j, j + 1);
			_(T, j, j + 1) = _(T, j + 1, j + 1);
			_(T, j + 1, j + 1) = t;
			if (j + 2 < n) {
				t = _(T, j, j + 2);
				_(T, j, j + 2) = _(T, j + 1, j + 2);
				_(T, j + 1, j + 2) = t;
			}
			swap_columns(B, j, j + 1);
		}
		if (_(T, j, j)) {
			tau = _(T, j + 1, j) / _(T, j, j);
			_(T, j + 1, j + 1) -= tau * _(T, j, j + 1);
			if (j + 2 < n) {
				_(T, j + 1, j + 2) -= tau * _(T, j, j + 2);
			}
			for (i = 0; i < p; i++) {
				_(B, i, j + 1) -= tau * _(B, i, j);
			}
		}
	}
	for (i = n - 1; i >= 0; i--) {
		for (k = 0; k < p; k++) {
			if (i < n - 1) {
				_(B, k, i) -= _(T, i, i + 1) * _(B, k, i + 1);
			}
			if (i < n - 2) {
				_(B, k, i) -= _(T, i, i + 2) * _(B, k, i + 2);
			}
			if (_(T, i, i) == 0) {
				return 1;
			}
			_(B, k, i) /= _(T, i, i);
		}
	}
	matrixf_transpose(B);
	return 0;
}

int matrixf_solve_spd(Matrixf* A, Matrixf* B)
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
			Bik = _(B, i, k);
			for (j = 0; j < i; j++) {
				Bik -= _(A, j, i) * _(B, j, k);
			}
			_(B, i, k) = Bik / _(A, i, i);
		}
		for (i = n - 1; i >= 0; i--) {
			Bik = _(B, i, k);
			for (j = i + 1; j < n; j++) {
				Bik -= _(A, i, j) * _(B, j, k);
			}
			_(B, i, k) = Bik / _(A, i, i);
		}
	}
	return 0;
}

int matrixf_solve_symm(Matrixf* A, Matrixf* B)
{
	int i, j, k, s = 0;
	const int n = A->size[0];
	const int p = B->size[1];
	float beta, tau, t;
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
				_(B, k, i) -= _(A, i, j) * _(B, k, j);
			}
		}
	}
	for (j = 0; j < n - 1; j++) {
		beta = _(A, j - s, j + 1);
		s = 0;
		if (fabsf(_(A, j, j)) < fabsf(beta)) {
			s = 1;
			t = _(A, j, j + 1);
			_(A, j, j + 1) = _(A, j + 1, j + 1);
			_(A, j + 1, j + 1) = t;
			if (j + 2 < n) {
				t = _(A, j, j + 2);
				_(A, j, j + 2) = _(A, j + 1, j + 2);
				_(A, j + 1, j + 2) = t;
			}
			t = _(A, j, j);
			_(A, j, j) = beta;
			beta = t;
			swap_columns(B, j, j + 1);
		}
		if (_(A, j, j)) {
			tau = beta / _(A, j, j);
			_(A, j + 1, j + 1) -= tau * _(A, j, j + 1);
			if (j + 2 < n) {
				_(A, j + 1, j + 2) -= tau * _(A, j, j + 2);
			}
			for (i = 0; i < p; i++) {
				_(B, i, j + 1) -= tau * _(B, i, j);
			}
		}
	}
	for (i = n - 1; i >= 0; i--) {
		for (k = 0; k < p; k++) {
			if (i < n - 1) {
				_(B, k, i) -= _(A, i, i + 1) * _(B, k, i + 1);
			}
			if (i < n - 2) {
				_(B, k, i) -= _(A, i, i + 2) * _(B, k, i + 2);
			}
			if (_(A, i, i) == 0) {
				return 1;
			}
			_(B, k, i) /= _(A, i, i);
		}
	}
	for (i = n - 1; i > 0; i--) {
		for (j = i + 1; j < n; j++) {
			for (k = 0; k < p; k++) {
				_(B, k, i) -= _(A, j, i) * _(B, k, j);
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
			Bik = _(B, i, k);
			for (j = 0; j < i; j++) {
				Bik -= _(A, i, j) * _(B, j, k);
			}
			_(B, i, k) = Bik;
		}
		for (i = n - 1; i >= 0; i--) {
			Bik = _(B, i, k);
			for (j = i + 1; j < n; j++) {
				Bik -= _(A, i, j) * _(B, j, k);
			}
			Aii = _(A, i, i);
			if (Aii == 0) {
				return 1;
			}
			_(B, i, k) = Bik / Aii;
		}
	}
	return 0;
}

int matrixf_solve_lsq(Matrixf* A, Matrixf* B)
{
	int i, j, k;
	const int m = A->size[0];
	const int n = A->size[1];
	const int p = B->size[1];
	float Aii, Bik;
	Matrixf X = { { n, p }, B->data };

	if ((m < n) || (B->size[0] != m)) {
		return -1;
	}
	matrixf_decomp_qr(A, 0, 0, B);
	for (k = 0; k < p; k++) {
		for (i = n - 1; i >= 0; i--) {
			Bik = _(B, i, k);
			for (j = i + 1; j < n; j++) {
				Bik -= _(A, i, j) * _(B, j, k);
			}
			Aii = _(A, i, i);
			if (Aii == 0) {
				return 1;
			}
			_(B, i, k) = Bik / Aii;
		}
		if (m != n && k > 0) {
			for (i = 0; i < n; i++) {
				_(&X, i, k) = _(B, i, k);
			}
		}
	}
	B->size[0] = n;
	return 0;
}

int matrixf_solve_qr(Matrixf* A, Matrixf* B, Matrixf* X, float tol)
{
	int i, j, k, rank;
	const int m = A->size[0];
	const int n = A->size[1];
	const int h = B->size[1];
	const int p = m < n ? m : n;
	float t, s, Bik;
	Matrixf perm = { { 1, n }, X->data };

	if (matrixf_decomp_qr(A, 0, &perm, B)) {
		return -1;
	}
	t = X->data[p - 1];
	s = _(A, p - 1, p - 1);
	for (j = 0; j < n; j++) {
		_(A, m - 1, j) = X->data[j];
	}
	_(A, p - 1, p - 1) = s;
	if (tol < 0) {
		tol = (m > n ? m : n) * EPS(_(A, 0, 0));
	}
	j = 0;
	while (j < p && fabsf(_(A, j, j)) > tol) {
		j++;
	}
	rank = j;
	for (k = 0; k < h; k++) {
		for (i = n - 1; i >= 0; i--) {
			if (i < rank) {
				Bik = _(B, i, k);
				for (j = i + 1; j < rank; j++) {
					Bik -= _(A, i, j) * _(X, j, k);
				}
				_(X, i, k) = Bik / _(A, i, i);
			}
			else {
				_(X, i, k) = 0;
			}
		}
	}
	for (j = 0; j < n; j++) {
		A->data[j] = _(A, m - 1, j);
	}
	A->data[p - 1] = t;
	perm.data = A->data;
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
		tol = q * EPS(_(A, 0, 0));
	}
	j = 0;
	while (j < p && fabsf(_(A, j, j)) > tol) {
		for (i = j + 1; i < m; i++) {
			_(A, i, j) = 0;
		}
		j++;
	}
	rank = j;
	matrixf_transpose(A);
	A->size[1] = rank;
	matrixf_decomp_qr(A, 0, 0, 0);
	for (k = 0; k < h; k++) {
		for (i = 0; i < n; i++) {
			if (i < rank) {
				Bik = _(B, i, k);
				for (j = 0; j < i; j++) {
					Bik -= _(A, j, i) * _(X, j, k);
				}
				_(X, i, k) = Bik / _(A, i, i);
			}
			else {
				_(X, i, k) = 0;
			}
		}
	}
	matrixf_accumulate_bwd(A, X);
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
		tol = q * EPS(get_norm2(&_(A, 0, 0), q, 1));
	}
	for (j = 0; j < p; j++) {
		sj = get_norm2(&_(A, 0, j), q, 1);
		if (sj > 0) {
			rj = sj > tol ? 1.0f / sj : 0.0f;
			for (i = 0; i < q; i++) {
				_(A, i, j) /= sj;
			}
			for (i = 0; i < n; i++) {
				_(X, i, j) *= rj;
			}
		}
	}
	matrixf_transpose(A);
	for (j = 0; j < q; j++) {
		for (i = 0; i < p; i++) {
			work[i] = _(A, i, j);
		}
		for (i = 0; i < p; i++) {
			for (Aij = 0, k = 0; k < p; k++) {
				Aij += work[k] * _(&V, i, k);
			}
			_(A, i, j) = Aij;
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
			t += fabsf(_(A, i, j));
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
				work[i] = _(&X, i, j);
			}
			for (i = 0; i < n; i++) {
				for (Xij = 0, r = 0; r < n; r++) {
					Xij += work[r] * _(A, i, r);
				}
				_(&X, i, j) = Xij;
			}
		}
		for (i = 0; i < n * n; i++) {
			N.data[i] += c * X.data[i];
			D.data[i] += c * X.data[i] * ((k % 2) ? -1 : 1);
		}
	}
	if (matrixf_solve_lu(&D, &N)) {
		return 1;
	}
	for (k = 0; k < z; k++) {
		matrixf_multiply(&N, &N, A, 1, 0, 0, 0);
		if (k < z - 1) {
			for (i = 0; i < n * n; i++) {
				N.data[i] = A->data[i];
			}
		}
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
					c = _(C, i, j);
					for (k = 0; k < p; k++) {
						c += alpha * _(A, k, i) * _(B, j, k);
					}
					_(C, i, j) = c;
				}
			}
		}
		else { // C = alpha * A' * B + beta * C
			if (m != A->size[1] || p != B->size[0] || n != B->size[1]) {
				return -1;
			}
			for (j = 0; j < n; j++) {
				for (i = 0; i < m; i++) {
					c = _(C, i, j);
					for (k = 0; k < p; k++) {
						c += alpha * _(A, k, i) * _(B, k, j);
					}
					_(C, i, j) = c;
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
					b = _(B, j, k);
					for (i = 0; i < m; i++) {
						_(C, i, j) += alpha * _(A, i, k) * b;
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
					b = _(B, k, j);
					for (i = 0; i < m; i++) {
						_(C, i, j) += alpha * _(A, i, k) * b;
					}
				}
			}
		}
	}
	return 0;
}