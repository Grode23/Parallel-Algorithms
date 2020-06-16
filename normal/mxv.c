# include <stdlib.h>
# include <stdio.h>
# include <time.h>

void main ( int argc, char *argv[] )
  
{
  double **a, *b, *c;
  int i, j, N;
 
 
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
	Assign values to the A and B matrices.
	*/
	srand ( time ( NULL));

	for ( i = 0; i < N; i++ ) 
		for ( j = 0; j < N; j++ )
	      		a[i][j] = ( double ) rand() / (RAND_MAX * 2.0 - 1.0);

	for ( i = 0; i < N; i++ ) {
	    b[i] = ( double ) rand() / (RAND_MAX * 2.0 - 1.0);
        c[i] = 0.0;
	}

	//Starting time of solution
	clock_t start = clock();

	/* computation */
	for ( i = 0; i < N; i++) {
		c[i] = 0.0;
		for ( j = 0; j < N; j++ )
			c[i] += a[i][j] * b[j];
	}

	//Finishing time of solution
	clock_t finish = clock();
	double time_spent = (double)(finish - start) / CLOCKS_PER_SEC;
 

	/* output of data */
	for ( i = 0; i < N; i++ ) {
		//for ( j = 0; j < N; j++ )
			//printf ("%1.3f ", a[i][j]); 
		printf("\t %1.3f ", b[i]);
		printf("\t %1.3f \n", c[i]);
	}

	printf("Time spent: %f\n", time_spent);
}


