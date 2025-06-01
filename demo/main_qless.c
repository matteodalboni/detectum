/* Computing the solution of the linear system "A*x = b" without
* explicitly forming the matrix Q (Q-less QR decomposition). In
* particular: A'*A*x = A'*b --> R'*Q'*Q*R*x = A'*b --> R'*R*x = A'*b.
* 
* Matlab code:
* rng default;
* A = gallery('randsvd', [5, 3], 1000);
* b = ones(size(A,1),1);
* bb = A'*b;
* R = qr(A);
* x = R\(R'\bb)
* y = A\b;
* norm(x - y)
* fileID = fopen("./A.bin",'w');
* fwrite(fileID, A, 'single');
* fclose(fileID);
*/

#include <stdio.h>
#include <stdlib.h>
#include "detectum.h"

int main()
{
	Matrixf A = matrixf(5, 3);
	Matrixf b = matrixf(A.size[0], 1);
	Matrixf x = matrixf(A.size[1], 1);
	FILE* A_file = fopen("../A.bin", "rb");
	int i;

	if (!A.data) return -1;
	fread(A.data, sizeof(float), (size_t)(A.size[0] * A.size[1]), A_file);
	fclose(A_file);
	printf("A = \n"); matrixf_print(&A, "%9.4f "); printf("\n");
	for (i = 0; i < b.size[0]; i++) at(&b, i, 0) = 1;
	printf("b = \n"); matrixf_print(&b, "%9.4f "); printf("\n");
	matrixf_multiply(&A, &b, &x, 1, 0, 1, 0);
	matrixf_decomp_qr(&A, 0, 0, 0);
	matrixf_transpose(&A); matrixf_solve_tril(&A, &x, &b, 0);
	matrixf_transpose(&A); matrixf_solve_triu(&A, &b, &x, 0);
	printf("x = \n"); matrixf_print(&x, "%9.4f "); printf("\n");

	free(A.data);
	free(b.data);
	free(x.data);

	return 0;
}