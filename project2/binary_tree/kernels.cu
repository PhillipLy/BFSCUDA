#include <stdio.h>
#include "kernels.cuh"



__global__ void build_binary_tree(int *x, int *child, int *root, unsigned int n)
{
	//Instantiate variables
	unsigned int bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = blockDim.x*gridDim.x;
	unsigned int offset = 0;
	bool newBody = true;
	int rootValue = *root;

	// build binary tree
	int childPath;
	int temp;
	offset = 0;
	while((bodyIndex + offset) < n){

		if(newBody){
			newBody = false;

			temp = 0;
			childPath = 0;
			if(x[bodyIndex + offset] > rootValue){
				childPath = 1;
			}
		}
		int childIndex = child[temp*2 + childPath];
	
		// traverse tree until we hit leaf node
		while(childIndex >= 0){
			temp = childIndex;
			childPath = 0;
			if(x[bodyIndex + offset] > temp){
				childPath = 1;
			}

			childIndex = child[2*temp + childPath];
		}


		if(childIndex != -2){
			int locked = temp*2 + childPath;
			if(atomicCAS(&child[locked], childIndex, -2) == childIndex){
				if(childIndex == -1){
					child[locked] = x[bodyIndex + offset];
				}

				offset += stride;
				newBody = true;
			}
		}

		__syncthreads(); // not strictly needed 
	}
}