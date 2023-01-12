# openMP_dense_mat_mult

a parallel program implemented using OpenMP, matmult_omp, that will take as input two matrices A 
and B, and will output their product.


Language used : C/C++
tech/lib used : OpenMP 


Parallelization methods:

Simple Parallelization of Matrix Multiplication: In this approach, we directly parallelize the serial code without any modification except for adding omp tags.
In this, we are dividing iterations of the outer for loop (please see code attached) into 
Multiple threads and each thread will do work for multiple rows of A and cols of B.
(We can use the reduction clause to aggregate sum for each thread if we declare sum outside the parallel zone).

Tiling Parallelization of Matrix Multiplication: In this approach, we divide A, B matrices into small tiles of specific size and calculate a small part of the result for each tile, aggregating the same to get the whole result. It will optimize the cache hit rate as we tile the matrices we get the required elements for calculations in the cache and don’t have to fetch the same from memory.
Tiles are divided between threads, and threads will perform the computation in parallel. 

Transpose Parallelization of Matrix Multiplication: In this approach, we take the transpose of B matrix and then do the calculations in parallel.
This will save time required for jumping to the start of each col.
Each thread will be responsible for multiplying multiple rows of A and B.