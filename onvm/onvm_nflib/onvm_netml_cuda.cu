#include <rte_common.h>
#include <rte_mbuf.h>
#include <rte_ip.h>
#include <rte_mempool.h>
#include <rte_cycles.h>
#include <rte_ring.h>
#include <rte_ethdev.h>
#include <rte_ether.h>
#include <rte_udp.h>

extern "C"{
  #include <rte_malloc.h>
  #include "onvm_netml.h"
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/* Function to help with nice error message */
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}



/* create a stream for data movement */
cudaStream_t data_stream;

/* this function initializes the data structure where we will get the data by page faults and store the numbers
 * This one also initializes the array that we will copy pkt_mbufs into */
extern "C"
void init_cuda_stream(){
  cudaStreamCreate(&data_stream);
}


__global__ void transfer_data(void **ptrs_from_pkts, int image_id, void * gpu_image_buffer){

  /* the ids for block and thread */
  uint16_t bid = blockIdx.x;
  uint16_t tid = threadIdx.x;

  data_struct *data_in_pkts = (data_struct *) (ptrs_from_pkts[bid]);
  int position = data_in_pkts->position;
  

  float * empty_buffer = (float *) gpu_image_buffer;
  if(tid<NUM_IN_PKTS){
    empty_buffer[position+tid] = ((float *)ptrs_from_pkts)[tid];
  }
}


// ONVM facing function to transfer to GPU
extern "C"
void transfer_to_gpu(void **ptrs_from_pkts, int image_id, int num_of_pointers){
  transfer_data<<<num_of_pointers, NUM_IN_PKTS>>>(ptrs_from_pkts, image_id, gpu_image_buffers[image_id]);
}


//initialize the files data structure...
extern "C"
void image_init(void){
  int i;
  //allocating memory for structs
  images = (image_data *) rte_malloc(NULL, sizeof(image_data)*MAX_GPU_IMAGE, 0);
  
  for(i = 0; i<MAX_GPU_IMAGE; i++){
    images[i].img_data_ptrs = (void **) rte_malloc(NULL, sizeof(void *)*NUM_OF_PKTS, 0);
    images[i].image_id = i;
    images[i].status = empty;
    images[i].num_data_points_stored = 0;
    images[i].image_data = (float *) rte_malloc(NULL, sizeof(float)*IMAGE_SIZE, 0);
  }
}


//record keeping NF