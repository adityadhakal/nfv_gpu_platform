#ifndef _ONVM_ML_LIBRARIES_H
#define _ONVM_ML_LIBRARIES_H
#include <cuda_runtime.h>

#define NUMBER_OF_MODELS (11)
#define MAX_IMAGES_BATCH_SIZE 64
typedef struct nflib_ml_fw_load_params_t {
	const char* file_path;
	uint32_t file_len;
	uint32_t load_options; //Tensorrt:1 load from manager, 0 load from IF
	void* model_handle;
	void *ml_file_buffer; //ml_file buffer
	size_t model_buffer_size; //ml file buffer size
}nflib_ml_fw_load_params_t;

typedef struct nflib_ml_fw_link_params_t {
	const char * file_path; //some models have one shot loading.. so we will load them while linking
	void* model_handle;
	int number_of_parameters;
	void* cuda_handles_for_gpu_data;
	void* gpu_side_input_pointer; //in platform where we cannot create GPU buffers easily, the platform can create it and populate the address here
	void* gpu_side_output_pointer;
	int link_options; //1- link the model, 0- do not link the modelss
	void *ml_file_buffer; //ml_file_buffer
	size_t model_buffer_size; //ml_model_buffer_size
}nflib_ml_fw_link_params_t;

typedef struct nflib_ml_fw_infer_params_t {
	void* model_handle;
	void* input_data;
	size_t input_size; //bytes value of input
	uint32_t num_packets[MAX_IMAGES_BATCH_SIZE]; //number of total packets... for translation
	void *array_of_packets[MAX_IMAGES_BATCH_SIZE];//array of packets array
	float* output;
	void* evaluation_time;
	cudaStream_t *stream;
	int cuda_event_flag;
	int batch_size;
	cudaHostFn_t callback_function;
	void *callback_data;
	uint32_t file_len;
	uint32_t load_options;
}nflib_ml_fw_infer_params_t;

//typedef nflib_aio_info_t nflib_ml_status_t;

/** Signature of Deep Learning/Machine Learning Framework Operations *
typedef int (*ml_fw_load_model)(nflib_ml_fw_load_params_t* load_params,  nflib_ml_status_t *status);
typedef int (*ml_fw_link_model)(nflib_ml_fw_link_params_t* load_params,  nflib_ml_status_t *status);
typedef int (*ml_fw_infer_batch)(nflib_ml_fw_infer_params_t* infer_params,  nflib_ml_status_t *status);
typedef int (*ml_fw_get_inf_res)(nflib_ml_fw_infer_params_t* infer_params,  nflib_ml_status_t *status);
typedef int (*ml_fw_deinit)(uint32_t options);
*/

typedef int (*ml_fw_load_model)(nflib_ml_fw_load_params_t* load_params,  void *status);
typedef int (*ml_fw_link_model)(nflib_ml_fw_link_params_t* link_params,  void *status);
typedef int (*ml_fw_infer_batch)(nflib_ml_fw_infer_params_t* infer_params,  void *status);
typedef int (*ml_fw_get_inf_res)(nflib_ml_fw_infer_params_t* infer_params,  void *status);
typedef int (*ml_fw_deinit)(uint32_t options);



/** Set of Deep Learning/Machine Learning Framework Operations to be registered by the NFs **/
typedef struct ml_framework_operations_t {
	ml_fw_load_model 	load_model_fptr;
	ml_fw_link_model 	link_model_fptr;
	ml_fw_infer_batch	infer_batch_fptr;
	ml_fw_get_inf_res	get_inf_res_fptr;
	ml_fw_deinit		deinit_fptr;
}ml_framework_operations_t;

/** API for the NF to register the Deep Learning/Machine Learning Framework specific Operations **/
int nflib_register_ml_fw_operations(ml_framework_operations_t *ops);


#endif
