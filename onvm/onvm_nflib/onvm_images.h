#ifndef _ONVM_IMAGES_H
#define _ONVM_IMAGES_H

#define NUM_IN_PKTS 92
#define NUM_SIZE 4 //4 bytes number
#define IMAGE_NUM_ELE 3*224*224
#define IMAGE_SIZE 3*224*224*4
#define NUM_OF_PKTS IMAGE_SIZE/(NUM_IN_PKTS*NUM_SIZE)
#define IMAGE_BATCH 256

#include <rte_hash.h>


typedef enum data_status{empty, occupied, ready} data_status; //empty... data can be filled, occupied: some pointers available, ready: ready to be processed i.e. data filled.
typedef struct data_struct{
  int file_id;
  int position;
  int number_of_elements;
  float data_array[NUM_IN_PKTS];
}data_struct;

void * gpu_image_buffers[MAX_IMAGE];

void transfer_to_gpu(void ** ptrs_from_pkts, int image_id, int num_of_pointers);

typedef struct image_data{
  int image_id;
  void ** img_data_ptrs;
  float image_data_arr[IMAGE_NUM_ELE]; //the entire data array, so the mempool will give us place to keep entire data
  int num_data_points_stored; //if this is same as IMAGE_NUM_ELE then we can process the image.
  data_status status; // is the buffer empty (original state), occupied (some data in it), ready (ready to be processed)
} image_data;


//void image_init(struct onvm_nf_info *nf, struct onvm_nf_info *original_nf);

/* count the current images pending */
void **image_buffers;

#endif
