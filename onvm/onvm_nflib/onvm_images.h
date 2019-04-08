#ifndef _ONVM_IMAGES_H
#define _ONVM_IMAGES_H

#define NUM_IN_PKTS 96
#define NUM_SIZE 4 //4 bytes number
#define IMAGE_NUM_ELE 3*224*224
#define IMAGE_SIZE 3*224*224*4
#define NUM_OF_PKTS IMAGE_SIZE/(NUM_IN_PKTS*NUM_SIZE)
#define IMAGE_BATCH 8
#define IMAGENET_OUTPUT_SIZE 1000
#define NF_IMAGE_STATS_PERIOD_MS 1 //1 ms to check the stats
#define NF_INFERENCE_PERIOD 10

//#include <rte_hash.h>
//#include <rte_timer.h>

typedef enum data_status{empty, occupied, ready} data_status; //empty... data can be filled, occupied: some pointers available, ready: ready to be processed i.e. data filled.
typedef struct data_struct{
  int file_id;
  int position;
  int number_of_elements;
  float data_array[NUM_IN_PKTS];
}data_struct;

//void * gpu_image_buffers[MAX_IMAGE];

void transfer_to_gpu(void ** ptrs_from_pkts, int image_id, int num_of_pointers);


typedef struct image_data{
  int image_id;
  void ** img_data_ptrs;
  int batch_size; //number of image in batch...
  float image_data_arr[IMAGE_BATCH*IMAGE_NUM_ELE]; //the entire data array, so the mempool will give us place to keep entire data
  float output[IMAGE_BATCH*IMAGENET_OUTPUT_SIZE]; //this will store the imagenet output for whole batch
  float stats[3]; //0 - time for memcpy+execution, 1- time for execution only,
  struct timespec timestamps[5];// 0- when all_data is ready 1 -when placed in GPU execution queue, 2- function execution started, data transfer 3-when the GPU processing started, 4 -when the callback returns
  int num_data_points_stored; //if this is same as IMAGE_NUM_ELE then we can process the image. can be used for batch too
  data_status status; // is the buffer empty (original state), occupied (some data in it), ready (ready to be processed)
} image_data;

//callback struct, for the function from GPU
  struct gpu_callback{
    struct onvm_nf_info *nf_info;
    struct image_data *ready_image;
  };


// a flag to say if the NF has to finish all the works...
int gpu_finish_work_flag;


//void image_init(struct onvm_nf_info *nf, struct onvm_nf_info *original_nf);

/* count the current images pending */
//TODO: Remove this variable.
//void **image_buffers; //this maybe unused... 

//the GPU "queue".
/* We have noticed that executing in CNTK is "asynchronous", i.e. we won't know when it ends.
 * So we need to have a GPU queue, that will store the information about the images that has been sent to GPU
 * for processing

 * we will keep the time spent to process information till we give back the image mempool. 
 * we need this array as this array should be private to the NF and no need to be shared to alternate NF
 * We will store the time in time_spec.. should suffice for microseconds
 */
int  gpu_queue_image_id[MAX_IMAGE];
//TODO: Remove the below array.. and make an array that will rather store the pair of image data and nf_info pointers
struct gpu_callback gpu_callbacks[MAX_IMAGE];
int num_elements_in_gpu_queue;
int gpu_queue_current_index;


#endif
