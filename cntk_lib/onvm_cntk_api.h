#ifndef ONVM_CNTK_API_H
#define ONVM_CNTK_API_H

#define NUMBER_OF_MODELS 6


/* This function loads the model file from disk
 * Arguments are:
 * model file path, reference to CPU side module pointer, reference to GPU side model pointer, flag 0 = load only on CPU, 1 = load only on GPU, 2 = load on both devices 
 */
int load_model(const wchar_t * file_path, void ** cpu_function_pointer, void ** gpu_function_pointer, int load_flag, int model_no);


/* this function extracts the pointers for the GPU side models if there are GPU side models loaded 
   the cudaIpcMemHandles_t array is also submitted as parameter which the function will update with the cuda Mem handles
*/
void get_cuda_pointers_handles(void * gpu_function_pointer, void * mem_handles);

/* information about how many GPU-side pointer, i.e. how many CUDA handles you have with this model */
int count_parameters(void * gpu_function_pointer);

/* get the number of inputs a model have */
int count_inputs(void *function_pointer);

/* actually get the dimension of the inputs */
void get_all_input_sizes(void * function_ptr, int *input_dim_array, int input_sequence);

/* function to attach the GPU pointers with the models */
int link_gpu_pointers(void * cpu_function_pointer, void * cuda_handles_for_gpu_data, int number_of_parameters);

/* evaluate a data set */
void evaluate_in_gpu_input_from_host(float *input, size_t input_size, float *output, void *function_pointer, void *evaluation_time, int cuda_event_flag, void * callback_function, void *callback_data);

/* evaluate an image in GPU */
void evaluate_image_in_gpu(void * image, void *function_pointer, void * gpu_callback_function, void * callback_data, int gpu_barrier_flag);


/*
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
*/

#endif
