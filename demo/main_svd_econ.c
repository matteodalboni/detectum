#define DETECTUM_ENABLE_PRINT
#include "detectum.h"

int main()
{
#if 0
#define m 5
#define n 4
  float A_data[] = {
    17, 24,  1,  8,
    23,  5,  7, 14,
     4,  6, 13, 20,
    10, 12, 19, 21,
    11, 18, 25,  2
  };
  Matrixf(U, m, n);
  Matrixf(V, n, n);
  Matrixf(US, m, n);
#else
#define m 4
#define n 5
  float A_data[] = {
    17, 24,  1,  8, 15,
    23,  5,  7, 14, 16,
     4,  6, 13, 20, 22,
    10, 12, 19, 21,  3
  };
  Matrixf(U, m, m);
  Matrixf(V, n, m);
  Matrixf(US, m, m);
#endif
  Matrixf(USVt, m, n);
  Matrixf A;
  int i;
  float A_data_copy[m * n] = { 0 };

  matrixf_init(&A, m, n, A_data, 1);
  for (i = 0; i < m * n; i++) A_data_copy[i] = A_data[i];
  printf("\nA =\n"); matrixf_print(&A, "%9.4f");
#if 1
  int exitflag = matrixf_decomp_svd(&A, &U, &V);
  printf("\nU =\n"); matrixf_print(&U, "%9.4f");
  printf("\nS =\n"); matrixf_print(&A, "%9.4f");
  printf("\nV =\n"); matrixf_print(&V, "%9.4f");
  matrixf_multiply(&U, &A, &US, 1.0f, 0.0f, 0, 0);
  matrixf_multiply(&US, &V, &USVt, 1.0f, 0.0f, 0, 1);
#else
  int exitflag;
  float sigma[m < n ? m : n];
  if (m >= n) {
    exitflag = matrixf_decomp_svd_jacobi(&A, 0, &V);
    for (i = 0; i < n; i++) {
      sigma[i] = normf(A.data + A.rows * i, A.rows, 1);
      printf("\nsigma[%d] = %9.4f", i, sigma[i]);
    }
    printf("\n\nU*S =\n"); matrixf_print(&A, "%9.4f");
    printf("\nV =\n"); matrixf_print(&V, "%9.4f");
    matrixf_multiply(&A, &V, &USVt, 1.0f, 0.0f, 0, 1);
  }
  else {
    exitflag = matrixf_decomp_svd_jacobi(&A, &U, 0);
    for (i = 0; i < m; i++) {
      sigma[i] = normf(A.data + i, A.cols, A.rows);
      printf("\nsigma[%d] = %9.4f", i, sigma[i]);
    }
    printf("\n\nU =\n"); matrixf_print(&U, "%9.4f");
    printf("\nS*V' =\n"); matrixf_print(&A, "%9.4f");
    matrixf_multiply(&U, &A, &USVt, 1.0f, 0.0f, 0, 0);
  }
#endif
  printf("\nU*S*V' =\n"); matrixf_print(&USVt, "%9.4f");
  for (i = 0; i < m * n; i++) A_data_copy[i] -= USVt.data[i];
  printf("\n||U*S*V' - A||_F = %g\n", normf(A_data_copy, m * n, 1));
  return exitflag;
}