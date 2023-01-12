/* assert */
#include <assert.h>

/* errno */
#include <errno.h>

/* fopen, fscanf, fprintf, fclose */
#include <stdio.h>

/* EXIT_SUCCESS, EXIT_FAILURE, malloc, free */
#include <stdlib.h>

//OpenMP header
#include <omp.h>

//Bool datatype
#include <stdbool.h>

static const double * CORRECT_RES_MATRIX;
static int NUM_THREAD;


static int create_mat(size_t const nrows, size_t const ncols, double ** const matp)
{
    double * mat=NULL;
    if (!(mat = (double*) malloc(nrows*ncols*sizeof(*mat)))) {
        goto cleanup;
    }

    /** Initialize matrix with random values **/
    for(size_t i = 0; i < nrows; i++){
        for (size_t j = 0; j < ncols; j++){
            mat[(i * ncols) + j] = (double)(rand() % 1000) / 353.0;
        }
    }
    /** End random initialization **/

    *matp = mat;

    return 0;

    cleanup:
    free(mat);
    return -1;
}

// Print matrix of row and col
// in 1D and 2D format
static void print_mat(const double* mat,size_t r,size_t c){
    printf("========= 1 D =========\n");
    for (size_t i=0; i<r*c; i++ )
        printf("%2f, ", mat[i]);
    printf("\n");  

    printf("========= 2 D =========\n");
    for (size_t i=0; i<r; i++ )  {      
        for (size_t j=0; j<c; j++)
            printf(" %2f\t", mat[(i*c)+j]) ;
        printf("\n");
    }
    printf("==================\n");
}


// Compare Two matrices with same num of  row and col
// return true/false
static bool compare_mat(double* C, const double* D, size_t r, size_t c){
    bool flag=true;
    for (int i=0;i<r*c;i++){
        if(C[i]!=D[i]) {
          printf(" index : %d, values : %2f\t%2f", i,C[i], D[i]) ;
          flag=false;
          break;
        }
    }

    if (!flag && (r*c < 100 )){
      print_mat(C,r,c);
      print_mat(D,r,c);
    }

    return flag;
}

// sets transpose of matrix MAT in tran_mat_ptr
static int transpose_mat(double const * const mat, double** tran_mat_ptr,size_t r,size_t c){

  double* tran_mat=NULL;
  if (!(tran_mat = (double*) malloc(r*c*sizeof(*tran_mat)))) {
        goto cleanup;
    }

    for(size_t i = 0; i < r; i++){
        for (size_t j = 0; j < c; j++){
            tran_mat[(j * r) + i] = mat[(i * c) + j];
        }
    }

    *tran_mat_ptr=tran_mat;
    return 0;

    cleanup:
    free(tran_mat);
    return -1;
}

// parallel matrix multiplication without changing program struct
static int mult_para(size_t const n, size_t const m, size_t const p,
                    double const * const A, double const * const B,
                    double ** const Cp)
{
  
  double time = omp_get_wtime(); // Timer
  int num_thrds=0;
  size_t i, j, k;
  double sum;
  char * is_equal_ser="TRUE";
  double * C = NULL;

  if (!(C = (double*) malloc(n*p*sizeof(*C)))) {
    goto cleanup;
  }

  omp_set_num_threads(NUM_THREAD);
  #pragma omp parallel 
  {
    num_thrds=omp_get_num_threads();
    #pragma omp parallel for collapse(2)
    for (size_t i=0; i<n; ++i) {
      for (size_t j=0; j<p; ++j) {
        double sum=0.0;
        for (size_t k=0; k<m; ++k) {
          sum += A[i*m+k] * B[k*p+j];
        }
        C[i*p+j] = sum;
      }
    }
  }

  *Cp = C;

  time=omp_get_wtime()-time; // Get EXEC Time 
  
  if(!compare_mat(C, CORRECT_RES_MATRIX, n, p)) // 
    is_equal_ser="FALSE";

  printf("%s | %ld | %ld | %ld | %ld | %d | %d | %s | %lf Sec|\n",__func__, n, m, m,p,NUM_THREAD,num_thrds,is_equal_ser,time);

  return 0;

  cleanup:
  free(C);

  /*failure:*/
  return -1;

}

// Serial matrix multiplication with transpose memory access optimization
static int mult_blocking(size_t const n, size_t const m, size_t const p,
                    double const * const A, double const * const B,
                    double ** const Cp)
{
  double * T = NULL;
  transpose_mat(B,&T,m,p);

  double time = omp_get_wtime(); // Timer
  int num_thrds=0;
  size_t i, j, k;
  double sum=0.0;
  char * is_equal_ser="TRUE";
  double * C = NULL;
  

  if (!(C = (double*) malloc(n*p*sizeof(*C)))) {
    goto cleanup;
  }

  omp_set_num_threads(NUM_THREAD);
  num_thrds=omp_get_num_threads();
  
  // a n*m b m*p  b`= t p*m
    for (i=0; i<n; ++i) {
      for (j=0; j<p; ++j) {
        sum=0.0;
        for (k=0; k<m; ++k) {
          sum += A[i*m+k] * T[j*m+k];
        }
        C[j*n+i] = sum;
      }
    }

  transpose_mat(C,&T,p,n);
  *Cp = T;

  time=omp_get_wtime()-time; // Get EXEC Time 
  
//   if(!compare_mat(T, CORRECT_RES_MATRIX, n, p)) // 
//     is_equal_ser="FALSE";

  printf("%s | %ld | %ld | %ld | %ld | %d | %d | REF | %lf Sec|\n",__func__, n, m, m,p,NUM_THREAD,num_thrds,time);

  return 0;

  cleanup:
  free(C);

  /*failure:*/
  return -1;
}

// Parallel matrix multiplication with transpose memory access optimization
static int mult_para_blocking(size_t const n, size_t const m, size_t const p,
                    double const * const A, double const * const B,
                    double ** const Cp)
{
  double * T = NULL;
  transpose_mat(B,&T,m,p);

  double time = omp_get_wtime(); // Timer
  int num_thrds=0;
  char * is_equal_ser="TRUE";
  double * C = NULL;
  

  if (!(C = (double*) malloc(n*p*sizeof(*C)))) {
    goto cleanup;
  }

  omp_set_num_threads(NUM_THREAD);
  #pragma omp parallel 
  {
    num_thrds=omp_get_num_threads();
    #pragma omp for 
    for (size_t i=0; i<n; ++i) {
      for (size_t j=0; j<p; ++j) {
        double sum=0.0;
        for (size_t k=0; k<m; ++k) {
          sum += A[i*m+k] * T[j*m+k];
        }
        C[j*n+i] = sum;
      }
    }
  }

  transpose_mat(C,&T,p,n);
  *Cp = T;

  time=omp_get_wtime()-time; // Get EXEC Time 
  
  if(!compare_mat(T, CORRECT_RES_MATRIX, n, p)) // 
    is_equal_ser="FALSE";

  printf("%s | %ld | %ld | %ld | %ld | %d | %d | %s | %lf Sec|\n",__func__, n, m, m,p,NUM_THREAD,num_thrds,is_equal_ser,time);

  return 0;

  cleanup:
  free(C);

  /*failure:*/
  return -1;
}

// get tile size for matrix multiplication with tiling
int get_tile_size(int n, int m, int p)
{
    int size=1;
    int sz=11;
    int size_arr[]={4,5,8,10,16,20,32,40,50,64,100};

    for (int i =0;i<sz;i++)
        if( n % size_arr[i] == 0 && m % size_arr[i] == 0 && p % size_arr[i] == 0)
            size=size_arr[i];
    return size;
}

// Serial matrix multiplication with tiling 
static int mult_tiling(size_t const n, size_t const m, size_t const p,
                    double const * const A, double const * const B,
                    double ** const Cp)
{

  double time = omp_get_wtime(); // Timer
  int num_thrds=1;
  char * is_equal_ser="TRUE";
  double * C = NULL;

  int tile_size = 1;
  tile_size=get_tile_size(n,m,p); // set tile size

  if (!(C = (double*) malloc(n*p*sizeof(*C)))) {
    goto cleanup;
  }

  omp_set_num_threads(NUM_THREAD);

    for (int i = 0; i < n; i += tile_size)
    {
        for (int j = 0; j < p; j += tile_size)
        {
        for (int k = 0; k < m; k += tile_size)
        {
            for (int i1 = i; i1 < i + tile_size; i1++)
            {
            for (int j1 = j; j1 < j + tile_size; j1++)
            {
                for (int k1 = k; k1 < k + tile_size; k1++)
                C[i1 * p + j1] += A[i1 * m + k1] * B[k1 * p + j1];
            }
            }
        }
        }
    }

  *Cp = C;


  time=omp_get_wtime()-time; // Get EXEC Time 

  
  if(!compare_mat(C, CORRECT_RES_MATRIX, n, p)) // 
    is_equal_ser="FALSE";

  printf("%s | %ld | %ld | %ld | %ld | %d | %d | %s | %lf Sec| Tile size : %d|\n",__func__, n, m, m,p,NUM_THREAD,num_thrds,is_equal_ser,time,tile_size);

  return 0;

  cleanup:
  free(C);

  /*failure:*/
  return -1;
}


static int mult_para_tiling(size_t const n, size_t const m, size_t const p,
                    double const * const A, double const * const B,
                    double ** const Cp)
{

  double time = omp_get_wtime(); // Timer
  int num_thrds=0;
  char * is_equal_ser="TRUE";
  double * C = NULL;

  int tile_size = 10;
  tile_size=get_tile_size(n,m,p); // set tile size

  if (!(C = (double*) malloc(n*p*sizeof(*C)))) {
    goto cleanup;
  }

  omp_set_num_threads(NUM_THREAD);

    #pragma omp parallel for
    for (int i = 0; i < n; i += tile_size)
    {
        if (omp_get_thread_num()==0) num_thrds=omp_get_num_threads(); // setting num_threads for report if in master thread

        for (int j = 0; j < p; j += tile_size)
        {
        for (int k = 0; k < m; k += tile_size)
        {
            for (int i1 = i; i1 < i + tile_size; i1++)
            {
            for (int j1 = j; j1 < j + tile_size; j1++)
            {
                for (int k1 = k; k1 < k + tile_size; k1++)
                {
                C[i1 * p + j1] += A[i1 * m + k1] * B[k1 * p + j1];
                }
            }
            }
        }
        }
    }

  *Cp = C;

  time=omp_get_wtime()-time; // Get EXEC Time 
  
  if(!compare_mat(C, CORRECT_RES_MATRIX, n, p)) // 
    is_equal_ser="FALSE";
  
  //reporting exec parameters
  printf("%s | %ld | %ld | %ld | %ld | %d | %d | %s | %lf Sec| Tile size : %d|\n",__func__, n, m, m,p,NUM_THREAD,num_thrds,is_equal_ser,time,tile_size);

  return 0;

  cleanup:
  free(C);

  /*failure:*/
  return -1;
}




int main(int argc, char * argv[])
{
  // size_t stored an unsigned integer
  size_t nrows, ncols, ncols2;
  double * A=NULL, * B=NULL, * C=NULL,* D=NULL, * E=NULL;

  if (argc != 5) {
    fprintf(stderr, "usage: matmult nrows ncols ncols2 num_threads\n");
    printf("argc : %d",argc);
    for (int i=0;i<argc;i++)
      printf("argv[%d] : %s",i,argv[i]);
    goto failure;
  }

  nrows = atoi(argv[1]);
  ncols = atoi(argv[2]);
  ncols2 = atoi(argv[3]);
  NUM_THREAD = atoi(argv[4]);

  if (create_mat(nrows, ncols, &A)) {
    perror("error");
    goto failure;
  }

  if (create_mat(ncols, ncols2, &B)) {
    perror("error");
    goto failure;
  }

  printf("FUNCTION NAME | ROW1 | COL1 | ROW2 | COL2 | NUM THREADS DEFINED | NUM THREADS ALLOCATED | RESULT IS_EQUAL SER | EXEC TIME|\n");

  if (mult_blocking(nrows, ncols, ncols2, A, B, &C)) {
    perror("error");
    goto failure;
  }
  CORRECT_RES_MATRIX=C;

  if (mult_para_blocking(nrows, ncols, ncols2, A, B, &D)) {
    perror("error");
    goto failure;
  }

  if (mult_tiling(nrows, ncols, ncols2, A, B, &D)) {
    perror("error");
    goto failure;
  }

  if (mult_para_tiling(nrows, ncols, ncols2, A, B, &D)) {
    perror("error");
    goto failure;
  }

  free(A);
  free(B);
  free(C);
  free(D);
  free(E);

  return EXIT_SUCCESS;

  failure:
  if(A){
    free(A);
  }
  if(B){
    free(B);
  }
  if(C){
    free(C);
  }
    if(D){
    free(D);
  }
  if(E){
    free(E);
  }
  return EXIT_FAILURE;
}
