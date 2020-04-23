#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "mpi.h"

void main ( int argc, char *argv[]) {
	double **a, *b, *c;
	int N;
	int rank, numtasks;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (argc != 2) {
		printf ("Usage : %s <matrix size>\n", argv[0]);
		exit(1);
	}
	N = strtol(argv[1], NULL, 10);

	/*
	   Allocate the matrices.
	 */
	a = ( double **)  malloc ( N * sizeof ( double *) );
	for ( i = 0; i < N; i++)
		a[i] = ( double * ) malloc ( N * sizeof ( double ) );
	b = ( double * ) malloc ( N * sizeof ( double ) );
	c = ( double * ) malloc ( N * sizeof ( double ) );
	/*
	   Assign values to the B and C matrices.
	 */
	srand ( time ( NULL));

	for (int i = 0; i < N; i++ )
		for (int j = 0; j < N; j++ )
			a[i][j] = 1.0;            //( double ) rand() / (RAND_MAX * 2.0 - 1.0);

	for (int i = 0; i < N; i++ ) {
		b[i] = 1.0;     //( double ) rand() / (RAND_MAX * 2.0 - 1.0);
		c[i] = 0.0;
	}

	/* computation */
	for ( i = 0; i < N; i++) {
		c[i] = 0.0;
		for ( j = 0; j < N; j++ )
			c[i] = c[i] + a[i][j] * b[j];
	}


	/* output of data -- master */


	for (int i = 0; i < N; i++ ) {
		for (int j = 0; j < N; j++ )
			printf ("%1.3f ", a[i][j]);
		printf("\t %1.3f ", b[i]);
		printf("\t %1.3f \n", c[i]);
	}

}
