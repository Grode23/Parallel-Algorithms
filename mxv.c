# include <stdlib.h>
# include <stdio.h>
# include <time.h>
# include <omp.h>

#define NUM_OF_THREADS 8

void main ( int argc, char *argv[] ) {
  double **a, *b, *c;
  int N;
 
 
	if (argc != 2) {
		printf ("Usage : %s <matrix size>\n", argv[0]);
		exit(1);
	}

	N = strtol(argv[1], NULL, 10);

	/*
	Allocate the matrices.
	*/
    a = ( double **)  malloc ( N * sizeof ( double *) );
	
	for (int i = 0; i < N; i++) 
		a[i] = ( double * ) malloc ( N * sizeof ( double ) );
	
	b = ( double * ) malloc ( N * sizeof ( double ) );
	c = ( double * ) malloc ( N * sizeof ( double ) );
	
	/*
	Assign values to the B and C matrices.
	*/
	srand ( time ( NULL));

	for (int i = 0; i < N; i++ ) 
		for (int j = 0; j < N; j++){
	      	a[i][j] = rand() / (RAND_MAX * 2.0 - 1.0);
		}


	for (int i = 0; i < N; i++ ) {
	    b[i] = rand() / (RAND_MAX * 2.0 - 1.0);
    }

    printf("Parallelism starts here\n");

	omp_set_num_threads(NUM_OF_THREADS);

	/* computation */
	#pragma omp parallel
	{
		double temp_c[N];

		int rank = omp_get_thread_num();
		printf("%d\n",rank );
		for (int i = rank; i < N; i+=NUM_OF_THREADS) {
			temp_c[i] = 0.0;
			for (int j = 0; j < N; j++ ){
				temp_c[i] += a[i][j] * b[j];
			}

		}
		#pragma omp critical
		{
			for(int i = 0; i < N; i++){
				if(temp_c[i] != 0){
					c[i] = temp_c[i];
				}
			}
		}    
	}	
	/* output of data -- master */
  	for (int i = 0; i < N; i++ ) {
		for (int j = 0; j < N; j++ )
			printf ("%1.3f ", a[i][j]); 
		printf("\t %1.3f ", b[i]);
		printf("\t %1.3f \n", c[i]);
	}

}


