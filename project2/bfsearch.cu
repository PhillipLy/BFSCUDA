#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#define MAX_THREADS_PER_BLOCK 512
#include "timer.cu"


int number_nodes;
int edge_list;
FILE *fp;

struct Node
{
	int starting;
	int ending;
	int no_of_edges;
};

// Kernel for Bidirectional BFS algorithm
__global__ void
Kernel_bfs( Node* graphNodes, int* graphEdges, bool* graphFrontier, bool* updatedFrontier, bool *visited, int* g_cost, int number_nodes)
{
	int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if( tid<number_nodes && graphFrontier[tid] )
	{
		graphFrontier[tid]=false;
		for( int i = graphNodes[tid].starting; i < (graphNodes[tid].no_of_edges + graphNodes[tid].starting); i++ )
			{
				int id = graphEdges[i];
				if(!visited[id])
				{
					g_cost[id] =  g_cost[tid] + 1;
					updatedFrontier[id] = true;
				}
			}
		for( int i = graphNodes[tid].ending; i > (graphNodes[tid].no_of_edges + graphNodes[tid].ending); i-- )
			{
				int id = graphEdges[i];
				if( !visited[id] )
				{
					g_cost[id] =  g_cost[tid] + 1;
					updatedFrontier[id] = true;
				}
			}
	}
}

__global__ void
Kernel_bfs2( bool* graphFrontier, bool *updatedFrontier, bool* visited, bool *g_over, int number_nodes)
{
	int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if( tid < number_nodes && updatedFrontier[tid] )
	{

		graphFrontier[tid] = true;
		visited[tid] = true;
		*g_over = true;
		updatedFrontier[tid] = false;
	}
}

void BFSGraph(int argc, char** argv);
// Main Implementation
int main( int argc, char** argv )
{
	number_nodes = 0;
	edge_list = 0;
	BFSGraph( argc, argv);
}

void Usage(int argc, char**argv)
{

fprintf(stderr,"Usage: %s <input_file>\n", argv[0]);

}
// BFS Implementation
void BFSGraph( int argc, char** argv )
{

	double timing, io_timing, traversing_time;
	int is_output_timing=1;
        char *input_f;
	if( argc!=2 ) {
	Usage(argc, argv);
	exit(0);
	}

	//set counter for io timing
	if (is_output_timing) io_timing = wtime();
	input_f = argv[1];
	printf("Reading File (In-progress)\n");
	//Process Graph by reading from a file
	fp = fopen(input_f,"r");
	if (!fp)
	{
		printf("Error Reading graph file\n");
		return;
	}

	int source = 0;

	fscanf(fp,"%d",&number_nodes);

	int block = 1;
	int num_of_threads_per_block = number_nodes;

	if( number_nodes>MAX_THREADS_PER_BLOCK )
	{
		block = (int)ceil(number_nodes/(double)MAX_THREADS_PER_BLOCK);
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
	}

	// host memory allocation
	Node* graphNodes_host = (Node*) malloc(sizeof(Node)*number_nodes);
	bool *track_graphFrontier = (bool*) malloc(sizeof(bool)*number_nodes);
	bool *updated_frontier = (bool*) malloc(sizeof(bool)*number_nodes);
	bool *visited_host = (bool*) malloc(sizeof(bool)*number_nodes);

	int start, end, edgeno;
	// Memory Initialization
	for( unsigned int i = 0; i < number_nodes; i++)
	{
		fscanf(fp,"%d %d",&start,&edgeno);
		graphNodes_host[i].starting = start;
		graphNodes_host[i].ending = end;
		graphNodes_host[i].no_of_edges = edgeno;
		track_graphFrontier[i]=false;
		updated_frontier[i]=false;
		visited_host[i]=false;
	}

	//process the source node by reading it from the file
	fscanf(fp,"%d",&source);
	source=0;

	//set the source node as true in the frontier
	track_graphFrontier[source]=true;
	visited_host[source]=true;

	fscanf(fp,"%d",&edge_list);

	int id,cost;
	int* graphEdge_host = (int*) malloc(sizeof(int)*edge_list);
	for( int i = 0; i < edge_list ; i++ )
	{
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
		graphEdge_host[i] = id;
	}

	if(fp)
		fclose(fp);

	printf("Finished Reading File\n");

	//set counter for execution time
	if (is_output_timing) {
			timing            = wtime();
			io_timing         = timing - io_timing;
			traversing_time = timing;
	}

	//Copy the Node list to device memory
	Node* graphNodes_device;
	cudaMalloc( (void**) &graphNodes_device, sizeof(Node)*number_nodes) ;
	cudaMemcpy( graphNodes_device, graphNodes_host, sizeof(Node)*number_nodes, cudaMemcpyHostToDevice) ;

	//Copy the Edge List to device Memory
	int* graphEdge_device;
	cudaMalloc( (void**) &graphEdge_device, sizeof(int)*edge_list) ;
	cudaMemcpy( graphEdge_device, graphEdge_host, sizeof(int)*edge_list, cudaMemcpyHostToDevice) ;

	//Copy the frontier to device memory
	bool* d_graph_frontier;
	cudaMalloc( (void**) &d_graph_frontier, sizeof(bool)*number_nodes) ;
	cudaMemcpy( d_graph_frontier, track_graphFrontier, sizeof(bool)*number_nodes, cudaMemcpyHostToDevice) ;

	bool* updated_frontier_device;
	cudaMalloc( (void**) &updated_frontier_device, sizeof(bool)*number_nodes) ;
	cudaMemcpy( updated_frontier_device, updated_frontier, sizeof(bool)*number_nodes, cudaMemcpyHostToDevice) ;

	//Copy the Visited nodes array to device memory
	bool* visited_device;
	cudaMalloc( (void**) &visited_device, sizeof(bool)*number_nodes) ;
	cudaMemcpy( visited_device, visited_host, sizeof(bool)*number_nodes, cudaMemcpyHostToDevice) ;

	// memory allocation for the result on host side
	int* h_cost = (int*) malloc( sizeof(int)*number_nodes);
	for( int i = 0; i < number_nodes; i++ )
		h_cost[i] =- 1;
	h_cost[source]=0;

	// device memory allocation for result
	int* d_cost;
	cudaMalloc( (void**) &d_cost, sizeof(int)*number_nodes);
	cudaMemcpy( d_cost, h_cost, sizeof(int)*number_nodes, cudaMemcpyHostToDevice) ;

	//instantiates a bool to check when execution is over
	bool *d_over;
	cudaMalloc( (void**) &d_over, sizeof(bool));

	printf("Copied Everything to GPU memory\n");

	// setup execution parameters
	dim3  grid( block, 1, 1);
	dim3  threads( num_of_threads_per_block, 1, 1);

	int k = 0;
	printf("Begin to traverse the tree\n");
	bool stop;
	//Perform kernel calls until all the elements of Frontier are not false
	do
	{
		//if no thread changes this value then the loop stops
		stop=false;
		cudaMemcpy( d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice) ;
		Kernel_bfs<<< grid, threads, 0 >>>( graphNodes_device, graphEdge_device, d_graph_frontier, updated_frontier_device, visited_device, d_cost, number_nodes);
		// check to see if kernel execution generated and error


		Kernel_bfs2<<< grid, threads, 0 >>>( d_graph_frontier, updated_frontier_device, visited_device, d_over, number_nodes);
		// check to see if kernel execution generated and error


		cudaMemcpy( &stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost) ;
		k++;
	}
	while(stop);


	printf("Kernel Executed %d times\n",k);

	// copy result from device to host
	cudaMemcpy( h_cost, d_cost, sizeof(int)*number_nodes, cudaMemcpyDeviceToHost) ;

	//Store processed result and output to a file called result.txt
	FILE *fpo = fopen("result.txt","w");
	for( int i = 0; i < number_nodes; i++ )
		fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
	fclose(fpo);
	printf("Result stored in result.txt\n");

	// free memory - clean up
	free( graphNodes_host);
	free( graphEdge_host);
	free( track_graphFrontier);
	free( updated_frontier);
	free( visited_host);
	free( h_cost);
	cudaFree(graphNodes_device);
	cudaFree(graphEdge_device);
	cudaFree(d_graph_frontier);
	cudaFree(updated_frontier_device);
	cudaFree(visited_device);
	cudaFree(d_cost);

	if (is_output_timing) {
            timing = wtime();
            traversing_time = timing - traversing_time;
        }

	if (is_output_timing) {
			io_timing += wtime() - timing;
			printf("\nPerforming **** Parallel BFS (CUDA version) ****\n");
			printf("I/O time           = %10.4f sec\n", io_timing);
			printf("Traversing timing = %10.4f sec\n", traversing_time);
	}
}
