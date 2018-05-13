#include <iostream>
#include <time.h>
#include <random>
#include "kernels.cuh"



int main()
{
	unsigned int n = 32;

	// variables instantiations
	int *h_x;
	int *d_x;
	int *h_root;
	int *d_root;
	int *h_child;
	int *d_child;

	// initiate memory allocation
	h_x = (int*)malloc(n*sizeof(int));
	h_root = (int*)malloc(sizeof(int));
	h_child = (int*)malloc(2*(n+1)*sizeof(int));
	cudaMalloc((void**)&d_root, sizeof(int));
	cudaMalloc((void**)&d_x, n*sizeof(int));
	cudaMalloc((void**)&d_child, 2*(n+1)*sizeof(int));
	cudaMemset(d_child, -1, 2*(n+1)*sizeof(int));


	// fill h_temp and h_x arrays
	for(unsigned int i = 0; i < n; i++){
		h_x[i] = i+1;
	}

	for(unsigned int i=0;i<n;i++){
		unsigned int j = random() % (n-i);
		int temp = h_x[i];
		h_x[i] = h_x[i+j];
		h_x[i+j] = temp;
	}
	*h_root = h_x[0];

	for(unsigned int i=0;i<n;i++){
		std::cout<<h_x[i]<<" ";
	}
	std::cout<<""<<std::endl;


	// copy data to device
	cudaMemcpy(d_root, h_root, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, h_x, n*sizeof(int), cudaMemcpyHostToDevice);


	// kernel call
	dim3 gridSize = 8;
	dim3 blockSize = 8;
	build_binary_tree<<< gridSize, blockSize>>>(d_x, d_child, d_root, n);


	// copy from device back to host
	cudaMemcpy(h_child, d_child, 2*(n+1)*sizeof(int), cudaMemcpyDeviceToHost);


	// print tree
	for(unsigned int i = 0; i < 2*(n+1); i++){
		std::cout<<h_child[i]<<" ";
	}
	std::cout<<""<<std::endl;

	// free memory
	free(h_x);
	free(h_root);
	free(h_child);
	cudaFree(d_x);
	cudaFree(d_root);
	cudaFree(d_child);
}