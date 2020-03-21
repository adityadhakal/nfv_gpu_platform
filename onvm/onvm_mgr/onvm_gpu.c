#include <stdio.h>
#include <stdlib.h>
#include <rte_malloc.h>

#include <zmq.h>
#include <assert.h>

#include "onvm_gpu.h"
#include "onvm_mgr.h"
#include "onvm_nf.h"
#include "onvm_netml.h"

#ifdef ONVM_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include "onvm_cntk_api.h"
#include "tensorrt_api.h"
#include "onvm_ml_libraries.h"


onvm_gpu_model_operational_range_t onvm_gpu_ml_model_profiler_data[ONVM_MAX_GPU_ML_MODELS];
onvm_gpu_ra_info_t gpu_ra_info = {.active_nfs=0, .gpu_ra_avail=MAX_GPU_OVERPRIVISION_VALUE, .gpu_ra_wtlst=0, .waitlisted_nfs=0};
gpu_ra_mgt_t gpu_ra_mgt = {.gpu_ra_info=&gpu_ra_info, .ra_status= {0,}};
//declaration of internal function
void load_gpu_model(struct gpu_file_listing *ml_file);
//void load_old_profiler_data(model_profiler_data *profiler_data);
void load_old_profiler_data(char * filename, int model_index);

cudaIpcMemHandle_t *input_memhandles;
cudaIpcMemHandle_t *output_memhandles;

static inline struct onvm_nf_info *shadow_nf(int);
static inline struct onvm_nf_info *shadow_nf(int instance_id) {
	//returns the info of shadow NF ID
	unsigned shadow_id = get_associated_active_or_standby_nf_id(instance_id);
	struct onvm_nf *cl;
	cl = &(nfs[shadow_id]);

	return cl->info;
}

/****************************************************************************************
 * 						MODEL LOADING FROM DISK AND RETRIEVAL
 ****************************************************************************************/

/* this function is called by the main so that the ML models can be loaded to manager */
void init_ml_models(void) {
	/* the directory where the model is put */
	const char *model_dir = "/home/adhak001/openNetVM-dev/ml_models/";
        //const char *model_dir = "/tmp/";

	/* the file name of ml models */
	const char *models[NUMBER_OF_MODELS];
	models[0] = "AlexNet_ImageNet_CNTK.model";
	//models[0] = "resnet50_batch64.trt";
	models[1] = "ResNet50_ImageNet_CNTK.model";
	models[2] = "VGG19_ImageNet_Caffe.model";
	models[3] = "ResNet152_ImageNet_CNTK.model";
	//models[4] = "Fast-RCNN_Pascal.model";
	models[4] = "vgg16_batch64.trt";
	//models[5] = "SRGAN.model";
	models[5] = "AlexNet_ImageNet_CNTK.model";
	models[6] = "resnet50_batch64.trt";
	models[7] = "alexnet_batch128.trt";
	//	models[8] = "resnet152_batch128.trt";
	models[8] = "mobilenet_batch64.trt";
	//models[8] = "super_resolution.trt";
	models[9] = "vgg19_batch64.trt";
	models[10] = "resnet34_batch128.trt";
	models[11] = "super_resolutions.trt";

	/*the file name of historical runtime data */
	const char *models_historical_dir = "/home/adhak001/openNetVM-dev/models_data/";
	const char *models_runtime[NUMBER_OF_MODELS];
	models_runtime[0] = "alexnet_cntk_runtime.txt";
	models_runtime[1] = "resnet50_cntk_runtime.txt";
	models_runtime[2] = "vgg19_cntk_runtime.txt";
	models_runtime[3] = "resnet152_cntk_runtime.txt";
	models_runtime[4] = "vgg16_tensorrt.txt";
	//models_runtime[4] = "fastrcnn_runtime.txt";
	//models_runtime[5] = "srgan_cntk_runtime.txt";
	models_runtime[5] = "alexnet_cntk_runtime.txt";
	models_runtime[6] = "resnet50_tensorrt.txt";
	models_runtime[7] = "alexnet_tensorrt.txt";
	models_runtime[8] = "resnet152_tensorrt.txt";
	models_runtime[9] = "vgg19_tensorrt.txt";
	models_runtime[10] = "resnet34_tensorrt.txt";
	models_runtime[11] = "vgg16_tensorrt.txt";
	/* platform type */
	ml_platform platforms[NUMBER_OF_MODELS];
	//platforms[0] = cntk;
	platforms[0] = tensorrt;
	platforms[1] = cntk;
	platforms[2] = cntk;
	platforms[3] = cntk;
	platforms[4] = tensorrt;
	platforms[5] = cntk;
	platforms[6] = tensorrt;
	platforms[7] = tensorrt;
	platforms[8] = tensorrt;
	platforms[9] = tensorrt;
	platforms[10] = tensorrt;
	platforms[11] = tensorrt;
	/* now after setting all that up, let's allocate one information at a time and fill that up */
	struct rte_mempool *ml_model_mempool = rte_mempool_lookup(_GPU_MODELS_POOL_NAME);


	/* now let's loop over mempool and get one model at a time and fill up the details as well as load it up */
	//struct gpu_file_listing *ml_file2 = (struct gpu_file_listing *)rte_malloc(NULL,sizeof(struct gpu_file_listing)*7, 0);
	int i;
	cuInit(0); //initializing GPU
	struct gpu_file_listing *ml_file;
	for(i = 0; i < NUMBER_OF_MODELS; i++) {
		//ml_file = &ml_file2[i];
		//ml_file = &ml_files[i];
		rte_mempool_get(ml_model_mempool, (void **)&ml_file);

		ml_file->model_info.platform = platforms[i];
		ml_file->model_info.file_index = i;
		/* copy the directory and file name */
		strncpy(ml_file->model_info.model_file_path, model_dir, strlen(model_dir));
		strncat(ml_file->model_info.model_file_path, models[i], strlen(models[i]));
		/* copy the path of the runtime data file */
		strncpy(ml_file->attributes.profile_data.file_path, models_historical_dir, strlen(models_historical_dir));
		strncat(ml_file->attributes.profile_data.file_path, models_runtime[i], strlen(models_runtime[i]));
		/* now we can send this model to be populated with the load model function */

		//if(i>6)
		load_gpu_model(ml_file);

		ml_files[i] = ml_file;

	}
	printf("Loaded all the ML files ..\n");

	input_memhandles = (cudaIpcMemHandle_t *)rte_malloc(NULL,sizeof(cudaIpcMemHandle_t)*(MAX_NFS),0);
	output_memhandles = (cudaIpcMemHandle_t *)rte_malloc(NULL,sizeof(cudaIpcMemHandle_t)*(MAX_NFS),0);
	cudaError_t cuda_return;
	/* create cudaMalloc for multiple NFs */
	for(i = 0; i<MAX_NFS; i++) {
		cuda_return = cudaMalloc(&gpu_side_input_buffer[i],SIZE_OF_AN_IMAGE_BYTES*MAX_IMAGES_BATCH_SIZE);
		if(cuda_return != cudaSuccess) {
			printf("Cannot malloc dev buffer for NF\n");
		}
		cuda_return = cudaIpcGetMemHandle(&input_memhandles[i],gpu_side_input_buffer[i]);
		if(cuda_return != cudaSuccess) {
			printf("Cannot get ipc handle for NF\n");
		}

		//now output side buffer
		cuda_return = cudaMalloc(&gpu_side_output_buffer[i], sizeof(float)*1000*MAX_IMAGES_BATCH_SIZE);
		if(cuda_return != cudaSuccess) {
			printf("Cannot malloc dev buffer for NF\n");
		}

		cuda_return = cudaIpcGetMemHandle(&output_memhandles[i],gpu_side_output_buffer[i]);
		if(cuda_return != cudaSuccess) {
			printf("Cannot get ipc handle for NF\n");
		}
		else
		printf("got IPC for output buffer \n");

	}

}

/* loads the model for the manager */
void load_gpu_model(struct gpu_file_listing *ml_file) {

	int flag = 1; //all models are loaded to GPU, flag = 0 cpu only, flag = 1 gpu only
	int num_of_parameters = 0;//the number of parameters of a GPU side function

	/* create load model parameters */
	nflib_ml_fw_load_params_t load_model_params;
	load_model_params.file_path = ml_file->model_info.model_file_path;
	load_model_params.load_options = 1; //gpu model

	void *aio = NULL;

	/* load the ml model onto GPU */
	struct timespec begin,end;
	clock_gettime(CLOCK_MONOTONIC, &begin);
	if(ml_file->model_info.platform == cntk) {

		cntk_load_model(&load_model_params, aio); //loads the model
		printf("model handle %p \n", load_model_params.model_handle);
		ml_file->gpu_handle = load_model_params.model_handle;
		ml_file->model_info.model_handles.number_of_parameters = cntk_count_parameters(ml_file->gpu_handle);
		//we have to have loading API here.
		clock_gettime(CLOCK_MONOTONIC, &end);

		double time_taken_to_load = (end.tv_sec-begin.tv_sec)*1000000.0+(end.tv_nsec-begin.tv_nsec)/1000.0;
		printf("----- Time taken to load model %d is %f microseconds on GPU \n",ml_file->model_info.file_index,time_taken_to_load);

		//Now time to get file attributes..
		//general attributes..

		if(flag== 1)//gpu model
		{
			/* getting cuda handles */
			cudaIpcMemHandle_t * mem_handles= (cudaIpcMemHandle_t *) rte_malloc(NULL, ((sizeof(cudaIpcMemHandle_t))*(ml_file->model_info.model_handles.number_of_parameters)),0);

			printf("Number of cuda handles made %d\n",ml_file->model_info.model_handles.number_of_parameters);
			cntk_get_cuda_pointers_handles(ml_file->gpu_handle,mem_handles);
			ml_file->model_info.model_handles.cuda_handles = mem_handles;
			printf("Memhandle pointer %p \n", ml_file->model_info.model_handles.cuda_handles);
			//ml_file->model_info.model_handles.cuda_handles = mem_handles;
		}
		//update the attribute for CPU side..
		if(flag == 0)//cpu model
		{
			ml_file->model_info.model_handles.number_of_parameters = num_of_parameters;
		}
	}
	if(ml_file->model_info.platform == tensorrt) {
		printf("Loading tensorrt model \n");
		ml_file->model_info.model_size = 0;
		if(ml_file->model_info.file_index == 6){

			/*special case for resnet */
			load_model_params.load_options = 1;
			load_model_params.ml_file_buffer = rte_malloc(NULL,124346720,512);
			if(load_model_params.ml_file_buffer)
				printf("Created file buffer successfully \n");
			else
				printf("Cannot create file buffer successfully\n");
			load_model_params.model_buffer_size = 124346720;
			ml_file->model_info.model_cpu_address = load_model_params.ml_file_buffer;
			ml_file->model_info.model_size = load_model_params.model_buffer_size;

			tensorrt_load_model(&load_model_params,aio);
		}
		if(ml_file->model_info.file_index == 9){

			/*special case for resnet */
			load_model_params.load_options = 1;
			load_model_params.ml_file_buffer = rte_malloc(NULL,629282648,512);
			if(load_model_params.ml_file_buffer)
				printf("Created file buffer successfully \n");
			else
				printf("Cannot create file buffer successfully\n");
			load_model_params.model_buffer_size = 629282648;
			ml_file->model_info.model_cpu_address = load_model_params.ml_file_buffer;
			ml_file->model_info.model_size = load_model_params.model_buffer_size;

			tensorrt_load_model(&load_model_params,aio);
	}
if(ml_file->model_info.file_index == 7){

			/*special case for resnet */
			load_model_params.load_options = 1;
			load_model_params.ml_file_buffer = rte_malloc(NULL,247054120,512);
			if(load_model_params.ml_file_buffer)
				printf("Created file buffer successfully \n");
			else
				printf("Cannot create file buffer successfully\n");
			load_model_params.model_buffer_size = 247054120;
			ml_file->model_info.model_cpu_address = load_model_params.ml_file_buffer;
			ml_file->model_info.model_size = load_model_params.model_buffer_size;

			tensorrt_load_model(&load_model_params,aio);
	}
	}
	// load the csv file for data
	load_old_profiler_data(ml_file->attributes.profile_data.file_path,ml_file->model_info.file_index);
}

void load_old_profiler_data(char * file_path,int model_index ) {//model_profiler_data *profiler_data) {
	FILE *fp;
	char * line = NULL;
	size_t len = 0;
	ssize_t read;
	char buffer[1025];

	size_t bytes;

	int number_of_lines = 0;

	/* open the runtime file */
	fp = fopen(file_path,"r");
	if (fp == NULL) {
		printf("Couldn't open %s\n", file_path);
		return;
	}
	printf("Opened file %s for model historical data\n",file_path);
	/* this file should be organized in following way */
	// optimal percentage, step, percentage_range...
	/* first count number of lines in the file */
	while((bytes=fread(buffer, 1, sizeof(buffer)-1, fp))) {
		//lastchar = buffer[bytes-1];
		for(char *c = buffer; (c = memchr(c, '\n', bytes-(c-buffer))); c++) {
			number_of_lines++;
		}
		//check the code here https://codereview.stackexchange.com/questions/156477/c-program-to-count-number-of-lines-in-a-file
	}
	//go back to beginning of the file
	rewind(fp);

	//now we know number of lines in the runtime data file, we need to allocate space for each data we collect...
	//currently we store the runtime latency and corresponding runtime percentages and number of SMSss.
	//the file looks like
	// sm, percentage, latency for now
	/*
	 profiler_data->number_of_values = number_of_lines;
	 profiler_data->num_of_sm = (int *)rte_malloc(NULL, sizeof(int)*profiler_data->number_of_values, 0);
	 profiler_data->runtime_percentages = (int *)rte_malloc(NULL, sizeof(int)*profiler_data->number_of_values, 0);
	 profiler_data->runtime_latency = (int *)rte_malloc(NULL, sizeof(int)*profiler_data->number_of_values, 0);

	 */

	//rather than reading the profiler data, we need to read it as the optimal value, step value and operational range
	//get the pointer to the operational range data
	onvm_gpu_model_operational_range_t *operational_range = &onvm_gpu_ml_model_profiler_data[model_index];
	memset(operational_range, 0, sizeof(*operational_range));
	const char * token;

	int i;
	for(i = 0; i<number_of_lines; i++) {
		if((read = getline(&line, &len, fp)) != -1) {
			token = strtok(line, ",");
			sscanf(token, "%"SCNd16, &(operational_range->optimal_value));
			//keep scanning the rest of the line
			//printf("Optimal value read %"PRIu16"\n", operational_range->optimal_value);

			token = strtok(NULL, ",");
			if(token != NULL)
			sscanf(token, "%"SCNd16, &(operational_range->step_value));

			token = strtok(NULL, ",");
			if(token != NULL)
			sscanf(token, "%"SCNd16, &(operational_range->operational_range.min));

			token = strtok(NULL, ",");
			if(token != NULL)
			sscanf(token, "%"SCNd16, &(operational_range->operational_range.max));
		}
	}
}

/* sends the model info for the NF */
void * provide_nf_with_model(int file_index) {

	//Now get all the attributes of the ML model we are checking
	return (void *) &(ml_files[file_index]->model_info);
}

/* sends the input gpu buffer to the NF */
void * provide_nf_with_input_gpu_buffer(int service_id) {
	return (void *) &(input_memhandles[service_id]);
}

/* sends the output gpu buffer to the NF */
void * provide_nf_with_output_gpu_buffer(int service_id) {
	return (void *) &(output_memhandles[service_id]);
}

//check if the alternate is not null

void restart_nf(struct onvm_nf_info *nf) {
	//get the PID and construct a zmsg and send it out to orchestartor
	printf("####____------#### restarting the NF %d \n", nf->pid);
	send_message_to_orchestrator(create_zmsg(&(nf->pid),1, zrestart));
}

//this function should be only called if the NF has been ordered to get ready, i.e. called by not active NF only
//in case of first NF, this will never get called...
/*
void nf_is_gpu_ready(struct onvm_nf_info *nf) {
	//TODO just print for now
	printf("NF instance %d is now ready to process packets \n", nf->instance_id);
	//find the alternate NF... and then send it a message to shutdown...
	struct onvm_nf_info *alt_nf = shadow_nf(nf->instance_id);
	inform_NF_of_pending_restart(alt_nf);//sent message to tell the logic to restart it.

	//now let's change the access to the ring... the alt NF should be put in paused state and this NF in running state.
	//we have to send the message to NF to perform these.
	//the wakeup mgr is automatically called for that..
	onvm_nf_send_msg(nf->instance_id, MSG_RESUME,0,NULL);

	//send to active NF to stop
	onvm_nf_send_msg(alt_nf->instance_id, MSG_STOP, 0, NULL);
	printf("DEBUG, Sent message to both NFs to stop/wakeup etc.. \n");
}
*/

/****************************************************************************************
 * 						GPU Resource Allocation Management and Scheduling
 ****************************************************************************************/
/* Helper functions for finding the GPU percent wise compute time ... computed by undergrads */
//int suggest_gpu_percentage(float request_rate, int gpu_model);
inline int onvm_gpu_adjust_nf_gpu_perecentage(struct onvm_nf_info *nf);
inline int onvm_gpu_set_gpu_percentage(struct onvm_nf_info *nf, uint16_t gpu_percent);
inline int onvm_gpu_check_any_readjustment(void);
inline int onvm_gpu_check_any_readjustment(void) {
	int i = 0;
	int flag = 0;
	for(i=0; i<MAX_NFS; i++) {
		if(nfs[i].info){
			if(nfs[i].info->over_provisioned_for_slo || nfs[i].info->under_provisioned_for_slo) {
			//change the NF whose request over-provision or under provision to readjustment
			gpu_ra_mgt.ra_status[i] =  GPU_RA_NEEDS_READJUSTMENT;
			flag = 1;
			}

		}
	}
	if(flag)
		return 0;

	return 1;
}

inline int onvm_gpu_set_gpu_percentage(struct onvm_nf_info *nf, uint16_t gpu_percent) {
	printf("the percentage available %"PRIu16" percentage to set %"PRIu16"\n",gpu_ra_mgt.gpu_ra_info->gpu_ra_avail, gpu_percent);

	nf->gpu_percentage = gpu_percent;
	gpu_ra_mgt.nf_gpu_ra_list[nf->instance_id] = gpu_percent;
	gpu_ra_mgt.gpu_ra_info->gpu_ra_avail -= gpu_percent;
	gpu_ra_mgt.gpu_ra_info->active_nfs++;
	gpu_ra_mgt.ra_status[nf->instance_id] = GPU_RA_IS_SET;
	return 0;
}
inline int onvm_gpu_set_wt_list_gpu_percentage(struct onvm_nf_info *nf, uint16_t gpu_percent) {
	nf->gpu_percentage = gpu_percent;
	gpu_ra_mgt.nf_gpu_ra_list[nf->instance_id] = gpu_percent;
	gpu_ra_mgt.gpu_ra_info->gpu_ra_wtlst+= gpu_percent;
	gpu_ra_mgt.gpu_ra_info->waitlisted_nfs++;
	gpu_ra_mgt.ra_status[nf->instance_id] = GPU_RA_IS_WAITLISTED;
	return 0;
}
inline int compute_current_gpu_ra_stats(uint8_t *num_act_nfs, uint16_t *gpu_ra_avl_pct) {
	int i = 0;
	uint8_t act_nfs_count=0;
	uint16_t gpu_ra_used=0;

	for(;i<MAX_NFS;i++) {
			if (((onvm_nf_is_valid(&nfs[i])))&& (nfs[i].info->gpu_percentage)) {
				act_nfs_count++;
				//if(gpu_ra_mgt.ra_status[i] != GPU_RA_NEEDS_READJUSTMENT){
					gpu_ra_used+=nfs[i].info->gpu_percentage; //DO not count the RA for the NFs slated for readjustment
				//}
			}
	}
	if(num_act_nfs) {
		*num_act_nfs=act_nfs_count;
	}
	if(gpu_ra_avl_pct) {
		*gpu_ra_avl_pct= ((MAX_GPU_OVERPRIVISION_VALUE < gpu_ra_used)?(0):(MAX_GPU_OVERPRIVISION_VALUE - gpu_ra_used));
	}
	return act_nfs_count;
}

//Note: Resource must be released/readjusted when NF terminates/quits or restarts.
inline int onvm_gpu_adjust_nf_gpu_perecentage(struct onvm_nf_info *nf) {

	/*
	 //For Low Priority: Just set to Minimum TODO: Should we enable low prio to use the full GPU when none are using-- yes. then this is not good!
	 if(0 == nf->gpu_priority) {
	 uint16_t underprovision_val = onvm_gpu_ml_model_profiler_data[nf->gpu_model].operational_range.min;
	 //can fit the NF resource within underprovision value
	 if(underprovision_val < gpu_ra_info.gpu_ra_val) {
	 //try to provide as much left over above the underprovision value;
	 nf->gpu_percentage = gpu_ra_info.gpu_ra_val;
	 gpu_ra_info.gpu_ra_val -= nf->gpu_percentage;//=0;
	 gpu_ra_info.active_nfs++;
	 } else {
	 nf->gpu_percentage = 0; //just waitlist this low priority NF.
	 }
	 return 0;
	 }
	 //High Priorty NFs

	 */
	if(!nf) return 0;
#if 0

#endif
	return 0;
}

/****************************************************************************************
 * 						GPU Resource Allocation and Management APIs
 ****************************************************************************************/
//Function to be called when NF termainates/killed/ or is transitioned to move out (special case move to pause state).
inline int onvm_gpu_release_gpu_percentage_for_nf(struct onvm_nf_info *nf) {
	if(!nf) return (0);
	if((GPU_RA_IS_SET != gpu_ra_mgt.ra_status[nf->instance_id]) && (GPU_RA_NEED_TO_RELINQUISH != gpu_ra_mgt.ra_status[nf->instance_id])) return 0;

	// release GPU from the NF and add back to the GPU RM Pool.
	gpu_ra_mgt.gpu_ra_info->gpu_ra_avail += nf->gpu_percentage;
	nf->gpu_percentage = 0;
	//gpu_ra_mgt.gpu_ra_info->active_nfs--;
	//gpu_ra_mgt.ra_status[nf->instance_id] = GPU_RA_NOT_SET;
	//gpu_ra_mgt.nf_gpu_ra_list[nf->instance_id] = 0;

	return 0;
}
/** API to get the NFs GPU % share: This will initially allocate 100% and oversubscribe to MAX (200%).
 Thereafter we need to reapportion GPU fairly amongst the contending NFs. (Policies: Uniform vs Rate vs cost vs Rate-cost proportional.)

 */
inline int onvm_gpu_get_gpu_percentage_for_nf(struct onvm_nf_info *nf) {
	if(!nf) return (0);

	//if the NF does not need GPU. We need to let it progress
	if(nf->gpu_model == 0)
		return 0;

	//check for valid gpu model
	if( (0>= nf->gpu_model) || (ONVM_MAX_GPU_ML_MODELS <= nf->gpu_model)){
		check_and_wakeup_nf(nf->instance_id);
		return (0);
	}

	//IF NF already has percentage set, then ignore the call :: Double check what should be done here.. not clear yet!
	if((nf->gpu_percentage) /*&& (GPU_RA_IS_SET == gpu_ra_mgt.ra_status[nf->instance_id])*/) {
		onvm_gpu_set_gpu_percentage(nf,nf->gpu_percentage);
		return 0;
	}

	//IF NF is marked for readjustment or is marked to relinquish its GPU resource then ignore.
	if(GPU_RA_NEEDS_READJUSTMENT == gpu_ra_mgt.ra_status[nf->instance_id] || GPU_RA_NEED_TO_RELINQUISH == gpu_ra_mgt.ra_status[nf->instance_id]) return 0;

	onvm_gpu_model_operational_range_t *gpu_ml_info = &(onvm_gpu_ml_model_profiler_data[nf->gpu_model]);
	//check if model has valid pre-profiled data
	nf->gpu_monitor_lat = (gpu_ml_info->optimal_value)?(0):(1);
	//set default as optimal value
	nf->gpu_percentage = (gpu_ml_info->optimal_value)?(gpu_ml_info->optimal_value):(DEFAULT_GPU_RA_VALUE);
	//set the GPU Resource Mgt state update
	gpu_ra_mgt.ra_status[nf->instance_id] = GPU_RA_NOT_SET;

	printf("This current NF's GPU resource status is %d\n",gpu_ra_mgt.ra_status[nf->instance_id]);

	/*
	 if(nf->gpu_percentage) {
	 //increment or decrement based on the run-time profiling results. TODO: handle this!
	 printf("NF Instance[%d] for model[%d] is already allocated:[%d], GPU (optimal=[%d], min=[%d], max=[%d]) RA(val=[%d], nfs=[%d]), \n",
	 nf->instance_id, nf->gpu_model, nf->gpu_percentage,
	 gpu_ml_info->optimal_value, gpu_ml_info->operational_range.min, gpu_ml_info->operational_range.max,
	 gpu_ra_info.gpu_ra_val, gpu_ra_info.active_nfs);
	 return 0;
	 
	 } else {
	 nf->gpu_percentage = gpu_ml_info->optimal_value;
	 }*/

	//Check current GPU status, whether it is underutilized (make sure to always allocate 100%+ of GPU resource)
	printf("The RA available is %d \n", gpu_ra_info.gpu_ra_avail);
	if(gpu_ra_info.gpu_ra_avail >= GPU_MAX_RA_PER_NF) {
		onvm_gpu_set_gpu_percentage(nf, GPU_MAX_RA_PER_NF);
		}
	// Check if request can be sufficiently met, with a step overprovision for NF?
	else if (/*(0 == nf->gpu_monitor_lat) && */gpu_ra_info.gpu_ra_avail > nf->gpu_percentage) {
		printf("NF's gpu percentage is %d \n",nf->gpu_percentage);
		//Space to over-provision the NFs GPU percentage.
		uint16_t overprovision_val = gpu_ml_info->operational_range.max;//onvm_gpu_ml_model_profiler_data[nf->gpu_model].operational_range.max;
		if(overprovision_val > gpu_ra_info.gpu_ra_avail) {
			onvm_gpu_set_gpu_percentage(nf, gpu_ra_info.gpu_ra_avail); //nf->gpu_percentage = gpu_ra_info.gpu_ra_val;
		} else {
			onvm_gpu_set_gpu_percentage(nf, overprovision_val); //nf->gpu_percentage = overprovision_val;
		}
	}
	// Check if GPU can satisfy step under-provision below knee for this model?
	else {
		uint16_t underprovision_val = gpu_ml_info->operational_range.min; //onvm_gpu_ml_model_profiler_data[nf->gpu_model].operational_range.min;
		//can fit the NF resource within underprovision value
		if(underprovision_val < gpu_ra_info.gpu_ra_avail) {
			//try to provide as much left over above the underprovision value;
			onvm_gpu_set_gpu_percentage(nf, gpu_ra_info.gpu_ra_avail);//nf->gpu_percentage = gpu_ra_info.gpu_ra_val;
		} else {

			//need to readjust resources of all NFs to accomodate this otherwise deny admission to this NF
			onvm_gpu_set_wt_list_gpu_percentage(nf, nf->gpu_percentage);
			gpu_ra_mgt.ra_status[nf->instance_id] = GPU_RA_NEEDS_ALLOCATION;

			//high priority: need to allocate and preempt low piority if any;
			if(nf->gpu_priority) {
				//high priority: need to allocate and preempt low piority if any;
			} else {
				nf->gpu_percentage = 0; //just waitlist this low priority NF.
				//gpu_ra_mgt.ra_status[nf->instance_id] = GPU_RA_IS_WAITLISTED;
			}
		}
	}
	//try to adjust the NFs GPU % based on the Global contention ( call only in timer callback when any NF is waiting for gpu ra needs allocation)
	//onvm_gpu_adjust_nf_gpu_perecentage(nf);

	printf("NF Instance[%d: Prio:%d] for model[%d] is allocated:[%d], GPU ( RA_status=[%d], optimal=[%d], min=[%d], max=[%d]) RA(val=[%d], nfs=[%d]), \n",
			nf->instance_id, nf->gpu_priority, nf->gpu_model, nf->gpu_percentage, gpu_ra_mgt.ra_status[nf->instance_id],
			gpu_ml_info->optimal_value, gpu_ml_info->operational_range.min, gpu_ml_info->operational_range.max,
			gpu_ra_info.gpu_ra_avail, gpu_ra_info.active_nfs);

	return (0);
}


//#define READJUSTMENT_DUE_TO_RATE
/* this function will be called peridically by a thread to re-allocate the GPU percentage for NFs */
int onvm_gpu_check_gpu_ra_mgt(void) {
	uint8_t act_nfs;
	uint16_t gpu_ra_available;
	int i = 0;

	//print out the current states for all NFs
	for (i=0;i<MAX_NFS;i++){
		if(nfs[i].info != NULL){
			printf("NF %d GPU Status: %d, NF status: %d, GPU percentage: %d, Priority %d \n",i,gpu_ra_mgt.ra_status[i],nfs[i].info->status, nfs[i].info->gpu_percentage, nfs[i].info->gpu_priority);
		}
	}
	printf("\n");

	//check the current GPU statistcs
		compute_current_gpu_ra_stats(&act_nfs, &gpu_ra_available);
	//Check the Phase, phase 1, is where the change is still permitted in GPU percentage for NFs,
		// Phase 1: should not have any NF relinquishing GPU . Phase 2: if there is even one GPU needs to be relinquishing

		//also check if there are any NF that needs readjustment or allocation
		uint8_t readjust_or_allocate = 0;
		for (i=0; i<MAX_NFS;i++)
		{
			if(gpu_ra_mgt.ra_status[i] == GPU_RA_NEED_TO_RELINQUISH){
				printf("In Phase 2. Wait for finishing the loading of model\n");
				return 0;
			}
			if((gpu_ra_mgt.ra_status[i] == GPU_RA_NEEDS_ALLOCATION) || (gpu_ra_mgt.ra_status[i] == GPU_RA_NEEDS_READJUSTMENT)  ){
				readjust_or_allocate++;
		}
	}

	//check if any NFs are waiting for RA
#ifdef READJUSTMENT_DUE_TO_RATE
	if(/*(0 == gpu_ra_mgt.gpu_ra_info->gpu_ra_wtlst)||*/(0 == onvm_gpu_check_any_readjustment())){
		printf("_+)_+_+_+++ Readjustment time \n");
		//we have an NF with recommended GPU percentage. Just change that NF to new GPU percentage
		for(i = 0; i<MAX_NFS;i++){
		uint8_t shadow_nf_id = get_associated_active_or_standby_nf_id(i);
		if(onvm_nf_is_valid(&nfs[shadow_nf_id]) && gpu_ra_mgt.ra_status[i]==GPU_RA_NEEDS_READJUSTMENT){

			printf("Am I valid %d  shadow %"PRIu8" percentages %"PRIu16" ?\n",i,shadow_nf_id, gpu_ra_mgt.nf_gpu_ra_list[i]);

			// Return the previous allocated GPU percentage
			onvm_gpu_release_gpu_percentage_for_nf(nfs[i].info);
			gpu_ra_mgt.ra_status[i] = GPU_RA_NEED_TO_RELINQUISH;

			//check leftover GPU percentage
			uint32_t leftover_gpu = gpu_ra_mgt.gpu_ra_info->gpu_ra_avail;
			uint32_t new_gpu_percent;
			if(leftover_gpu != 0)
				new_gpu_percent = MIN(nfs[i].info->recommended_gpu_percentage,leftover_gpu);
			else
				new_gpu_percent = nfs[i].info->recommended_gpu_percentage;
			//Now set it for new one
			onvm_gpu_set_gpu_percentage(nfs[shadow_nf_id].info,new_gpu_percent);

			gpu_ra_mgt.nf_gpu_ra_list[i] = MAX_GPU_OVERPRIVISION_VALUE+1;

			struct timespec time_we_send_msg;
			clock_gettime(CLOCK_MONOTONIC, &time_we_send_msg);
			long msg_sent_time = time_we_send_msg.tv_sec*1000000000+time_we_send_msg.tv_nsec;
			printf("The time we sent msg to shadow NF %ld\n",msg_sent_time);
			get_shadow_NF_ready(nfs[i].info);
		}
	}
		return 0;
	}
	else
		return 0;
#endif //readjustment_due_to_rates



	printf("GPU Resource available %d (percent), Number of NFs currently using GPU %"PRIu8"\n",gpu_ra_available, act_nfs);



	printf("The number of NFs we need to re-adjust or reallocate %d\n", readjust_or_allocate);

	//now if nobody needs to allocate or readjust and if the resources are all utilized then return
	if(!gpu_ra_available && !readjust_or_allocate)
		return 0;


	uint8_t needs_ra=0, readj_ra=0, set_ra=0, gpu_using_nfs=0, high_priority_nfs=0, low_priority_nfs=0, sum_h_p_knee = 0, sum_l_p_knee = 0, waitlisted_ra = 0;
	uint8_t nfs_need_ra_list[MAX_NFS], nfs_readj_ra_list[MAX_NFS], nfs_set_ra_list[MAX_NFS], knee_values[MAX_NFS], high_priority_nf_list[MAX_NFS], low_priority_nf_list[MAX_NFS], wait_listed_nf_list[MAX_NFS];
	uint8_t sum_of_all_knees = 0;

	//intialize the values of the variables to zero 
	for(i = 0; i<MAX_NFS; i++){
		nfs_need_ra_list[i] = 0;
		nfs_readj_ra_list[i] = 0;
		nfs_set_ra_list[i] = 0;
		knee_values[i] = 0;
		gpu_ra_mgt.nf_gpu_ra_list[i] = 0;
		high_priority_nf_list[i] = 0;
		low_priority_nf_list[i]= 0;
	}


	//count the num of NFs that need GPU RA, RJ and SET and see what the Knee for each NFs is
	//also learn about high and low priority NFs.
for(i=0; i<MAX_NFS; i++) {
		//store the knee values of all GPU running NF
	if(onvm_nf_is_valid(&nfs[i])){
		if((GPU_RA_NEEDS_ALLOCATION == gpu_ra_mgt.ra_status[i]) || (GPU_RA_IS_WAITLISTED == gpu_ra_mgt.ra_status[i])) {
			nfs_need_ra_list[needs_ra] = i;
			needs_ra+=1;
			knee_values[i] = onvm_gpu_ml_model_profiler_data[nfs[i].info->gpu_model].optimal_value;

		}
		else if ((GPU_RA_NEEDS_READJUSTMENT == gpu_ra_mgt.ra_status[i])) {
			nfs_readj_ra_list[readj_ra] = i;
			readj_ra+=1;
			//store the knee values too
			knee_values[i] = onvm_gpu_ml_model_profiler_data[nfs[i].info->gpu_model].optimal_value;
		}
		else if ((GPU_RA_IS_SET  == gpu_ra_mgt.ra_status[i])){
			nfs_set_ra_list[set_ra] = i;
			set_ra+=1;
			//store the knee values too
			knee_values[i] = onvm_gpu_ml_model_profiler_data[nfs[i].info->gpu_model].optimal_value;
		}

		//now check if NF is high priority or low priority
		if(nfs[i].info->gpu_priority){
			high_priority_nf_list[high_priority_nfs]=i;
			high_priority_nfs++;
			sum_h_p_knee += knee_values[i];
		}
		else{
			//Low priority NF
			low_priority_nf_list[low_priority_nfs]=i;
			low_priority_nfs++;
			sum_l_p_knee += knee_values[i];
		}

	}
}
	sum_of_all_knees = sum_l_p_knee+sum_h_p_knee;
	gpu_using_nfs = needs_ra+readj_ra+set_ra; //all NFs that are concerned about GPU resource

	printf("--------- Number of NFs that need allocation %"PRIu8" AND that needs readjustment %"PRIu8"\n",needs_ra, readj_ra);

	//TODO: Must have logic to track, find optimal Rate-cost proportional share for each NF and then trigger restart NFs for all that have changed GPU RA Profile.
	//The algorithm to do priority based proportional fair allocation
	/**
	 * Let N be the number of NFs running
	 * Find the number of higher priority NFs
	 * provide knee to them
	 * Then see leftover GPU. Provide that to low priority NFs
	 * if still there is more GPU left, the higher priority ones should get it proportionally
	 *
	 * Divide the total GPU with N. Getting share of each NF
	 * Check what GPU %  each NF has..  if that NF is "set" and has not requested for more GPU, we should only give it the lower value
	 * populate the value to the NF's secondary
	 * Recompute the remaining GPU.. perform the same water falling exercise again.
	 * i.e. provide the value for lowest wanting NF
	 * then, for all the NFs that has optimum value more than what our algorithm have, provide it to the NFs.
	 *
	 * Be careful, this is still phase 1, committing to phase 2 comes with additional checking.
	 */

	
	int gpu_percentage_in_play = (int) MAX_GPU_OVERPRIVISION_VALUE; //all the resource we can allocate.
	
	printf("Number of NFs using gpu %d set_ra %d \n",gpu_using_nfs, set_ra);

	// If the resource is divided into NFs equally
	if(gpu_using_nfs == 0)
		return 0;

	//the algorithm for max-min fairness

	//algorithm here...
	//first see if we can give the knee to all NFs
	gpu_percentage_in_play = MAX_GPU_OVERPRIVISION_VALUE - sum_of_all_knees;
	printf("GPU Percentage left beyond all Knees %d, high priority nf demand %d, low priority demand %d \n", gpu_percentage_in_play,sum_h_p_knee,sum_l_p_knee);
	//if positive... we can allocate Knee for everyone
	//then we can share rest proportionally
	//
	//if zero... we can allocate the Knee to everyone
	//
	//if negative, then we do not have enough resource, we will just proportionally reduce the resource from the knee

	//if we can suffice the knee for everyone, provide it.
	printf("before comparing gpu_percentage in play\n");
	if(gpu_percentage_in_play==0){
		printf("All NFs get their knee\n");
		//allocate the knee to everyone
		for(i = 0; i<MAX_NFS; i++){
			gpu_ra_mgt.nf_gpu_ra_list[i] = knee_values[i];
		}
	}

	//if we have more GPU to give around
	if(gpu_percentage_in_play > 0){
		printf("More than enough GPU\n");
		int percentage_left = gpu_percentage_in_play;
		//first allocate the knee then add the remaining percentage proportionally to higher priority NFs
		for(i = 0; i<high_priority_nfs; i++){
			gpu_ra_mgt.nf_gpu_ra_list[high_priority_nf_list[i]] = knee_values[high_priority_nf_list[i]]+(uint8_t) (knee_values[high_priority_nf_list[i]]*gpu_percentage_in_play/sum_h_p_knee);
			percentage_left = 0; //all leftover will be given to the high priority NFs.
		}
		//Now assign the knees to low priority NF.. Here we are sure that the low priority NFs will also get their knee.
		for(i = 0; i<low_priority_nfs; i++){
			if(sum_l_p_knee > 0)
				gpu_ra_mgt.nf_gpu_ra_list[low_priority_nf_list[i]] = knee_values[low_priority_nf_list[i]]+(uint8_t)(knee_values[low_priority_nf_list[i]]*percentage_left/sum_l_p_knee);
		}
	}

	//now we do not have enough GPU to give around
	if(gpu_percentage_in_play < 0){
		// First, do we have enough for all the high priority ones?
		// if yes, then allocate the resources for the high priorities ones
		int leftover_gpu = 0;
		if(sum_h_p_knee <= MAX_GPU_OVERPRIVISION_VALUE){
			leftover_gpu = MAX_GPU_OVERPRIVISION_VALUE-sum_h_p_knee;
			for(i = 0; i<high_priority_nfs; i++){
				gpu_ra_mgt.nf_gpu_ra_list[high_priority_nf_list[i]] = knee_values[high_priority_nf_list[i]];
			}

			//now see that low priority NFs get the leftover GPU divided proportionally compared to their knee
			//Now assign the knees to low priority NF.. Here we are sure that the low priority NFs will also get their knee.
			for(i = 0; i<low_priority_nfs; i++){
				if(sum_l_p_knee >0)
					gpu_ra_mgt.nf_gpu_ra_list[low_priority_nf_list[i]] = (knee_values[low_priority_nf_list[i]]*leftover_gpu/sum_l_p_knee);
			}
		}
		else if(sum_h_p_knee > MAX_GPU_OVERPRIVISION_VALUE){
			//in this case all we can do is proportionally allocate the GPU resources to the high priority nfs. we have to make the NFs that will be low priority wait.
			for(i = 0; i<high_priority_nfs; i++){
				gpu_ra_mgt.nf_gpu_ra_list[high_priority_nf_list[i]] = knee_values[high_priority_nf_list[i]]+(uint8_t) (knee_values[high_priority_nf_list[i]]*MAX_GPU_OVERPRIVISION_VALUE/sum_h_p_knee);
			}
			for(i = 0; i<low_priority_nfs; i++){
				gpu_ra_mgt.ra_status[low_priority_nf_list[i]] = GPU_RA_IS_WAITLISTED;
				wait_listed_nf_list[waitlisted_ra] = low_priority_nf_list[i];
				waitlisted_ra++;

			}
		}
	}

	for(i=0; i<needs_ra; i++) {
		printf("Needs GPU Allocation: %d, %d, %d\n", i, nfs_need_ra_list[i], nfs_need_ra_list[i]);
	}
	for(i=0; i<readj_ra; i++) {
		printf("Needs GPU Readjustment: %d, %d, %d\n", i, nfs_readj_ra_list[i], nfs_readj_ra_list[i]);
	}
	for(i=0; i<set_ra;i++){
		printf("GPU already set: %d, %d, %d\n", i, nfs_set_ra_list[i],nfs_set_ra_list[i]);
	}
	for(i=0; i<waitlisted_ra;i++){
		printf("GPU Waitlisted: %d\n", wait_listed_nf_list[i]);
	}

	for(i=0; i<MAX_NFS; i++){
		printf(" Knee Values %d %"PRIu8" RA list value %"PRIu16" ",i,knee_values[i],gpu_ra_mgt.nf_gpu_ra_list[i]);
	}
	printf("\n");

	//get the shadow NF of NFs that we decide to change "ready", i.e. send the messages.
	//this has to be done first because we have to "return" the GPU resources so we can provide it to NFs which need GPU percentages
	for(i=0; i<MAX_NFS; i++){
		if(gpu_ra_mgt.nf_gpu_ra_list[i]){
			//condition for waking up the shadow NF
			 if(gpu_ra_mgt.ra_status[i] == GPU_RA_IS_SET && gpu_ra_mgt.ra_status[get_associated_active_or_standby_nf_id(i)]==GPU_RA_NOT_SET){
				//printf("888888 ---------------- Marker to send message to %d NF\n", get_associated_active_or_standby_nf_id(i));
				//check if the percentage is same and if it is do not restart.
				uint8_t shadow_nf_id = get_associated_active_or_standby_nf_id(i);
				if(onvm_nf_is_valid(&nfs[shadow_nf_id])){

					printf("Am I valid %d  shadow %"PRIu8" percentages %"PRIu16" ?\n",i,shadow_nf_id, gpu_ra_mgt.nf_gpu_ra_list[i]);

					// Return the previous allocated GPU percentage
					onvm_gpu_release_gpu_percentage_for_nf(nfs[i].info);
					gpu_ra_mgt.ra_status[i] = GPU_RA_NEED_TO_RELINQUISH;
					//Now set it for new one
					onvm_gpu_set_gpu_percentage(nfs[shadow_nf_id].info,gpu_ra_mgt.nf_gpu_ra_list[i]);

					gpu_ra_mgt.nf_gpu_ra_list[i] = MAX_GPU_OVERPRIVISION_VALUE+1;

					struct timespec time_we_send_msg;
					clock_gettime(CLOCK_MONOTONIC, &time_we_send_msg);
					long msg_sent_time = time_we_send_msg.tv_sec*1000000000+time_we_send_msg.tv_nsec;
					printf("The time we sent msg to shadow NF %ld\n",msg_sent_time);
					get_shadow_NF_ready(nfs[i].info);
				}
				else
					printf("WARNING: Secondary NF not found running, cannot change GPU Percentage for NF ID %d\n",i);

			}
		}
	}

	//Now just provide percentages for NFs waiting allocation
	for(i=0; i<MAX_NFS; i++){
			if(gpu_ra_mgt.nf_gpu_ra_list[i]){
				//change the alternate's GPU percentage, and then send the messages... also put the NFs that are being messaged into "READJUSTING" state
				//printf("GPU RA List %d\n", i);
				if(gpu_ra_mgt.ra_status[i]==GPU_RA_NEEDS_ALLOCATION)
				{
					onvm_gpu_set_gpu_percentage(nfs[i].info,gpu_ra_mgt.nf_gpu_ra_list[i] );
					//gpu_ra_mgt.nf_gpu_ra_list[i] = MAX_GPU_OVERPRIVISION_VALUE+1;
					nfs[i].info->ring_flag = 1;
					check_and_wakeup_nf(i);
				}
			}
	}
	return 0;
}

int update_gpu_ra_status_ring_flag(struct onvm_nf_info *nf){
	uint8_t nf_id = nf->instance_id;
	uint8_t alt_nf_id = get_associated_active_or_standby_nf_id(nf_id);

	if(gpu_ra_mgt.ra_status[alt_nf_id] == GPU_RA_NEED_TO_RELINQUISH){
		//set the new NF GPU RA to set
		gpu_ra_mgt.ra_status[nf_id] = GPU_RA_IS_SET;

		//set the previous primary NF to relinquish
		gpu_ra_mgt.ra_status[alt_nf_id] = GPU_RA_NOT_SET;

		//change the ring flags.
		struct timespec current_time;
		clock_gettime(CLOCK_MONOTONIC, &current_time);
		long curr_time = current_time.tv_sec*1000000000+ current_time.tv_nsec;
		printf("Changing the ring flags from NFs Time now: %ld\n", curr_time);
		nfs[alt_nf_id].info->ring_flag = 0;
		nfs[alt_nf_id].info->gpu_percentage = 0;

		struct timespec time_we_send_msg;
		clock_gettime(CLOCK_MONOTONIC, &time_we_send_msg);
		long msg_sent_time = time_we_send_msg.tv_sec*1000000000+time_we_send_msg.tv_nsec;
		printf("The time we changed ring flag to shadow NF %ld\n",msg_sent_time);


		//sleep(5);//checking if sleep helps
		nfs[nf_id].info->ring_flag = 1;
		onvm_nf_send_msg(alt_nf_id,MSG_STOP,0,NULL);

		//gpu_ra_mgt.nf_gpu_ra_list[alt_nf_id] = 0; //clearing the lock on Phase 2
	}

	//start procedure to restart the other NF


	return 0;
}


int gpu_state_and_percentage_check(struct onvm_nf_info *nf){
	//check if the NF has GPU percentage or not... if not, stop the NF till it gets GPU percentage.

		check_and_block_nf(nf->instance_id);
		return 1;

}
/****************************************************************************************
 * 						NF Orchestrator specific functions
 ****************************************************************************************/
/*let the NF know that it is to be restarted.. similarly, let the other NF know that it should load the model in GPU and run dummy data */
void inform_NF_of_pending_restart(struct onvm_nf_info *nf) {
	onvm_nf_send_msg(nf->instance_id, MSG_RESTART, 0, NULL);
}

/* the function to send message to shadow NF */
void get_shadow_NF_ready(struct onvm_nf_info *shadow) {
	//no need to send the GPU percentage with the message as the GPU percentage will be written by the manager.
	uint8_t shadow_nf_id = get_associated_active_or_standby_nf_id(shadow->instance_id);

	if(&nfs[shadow_nf_id].info->gpu_percentage) {
		//get the NF to running and send the message to the NF to get GPU Ready... Till now the NF's ring access flag should be 0.
		nfs[shadow_nf_id].info->status = NF_RUNNING;
		onvm_nf_send_msg(shadow_nf_id, MSG_GET_GPU_READY, 0, NULL);
	}
	else
	{
		printf("Alternate NF does not have GPU percentage set so \"ready\" message not passed \n");
	}

}

#ifdef ONVM_GPU_TEST
void voluntary_restart_the_nf(struct onvm_nf_info *nf) {
	//check if alternate is active... if not cancel the volutary restart
	struct onvm_nf_info *alternate_nf = shadow_nf(nf->instance_id);
	if(!onvm_nf_is_valid(&nfs[alternate_nf->instance_id])) {
		printf("The alternate NF is not running.. cannot initiate voluntary restart \n");
	}
	else
	{
		get_shadow_NF_ready(nfs->info);
	}

}
#endif

/* ------- <MESASAGING API > ******** */
void init_zmq(void) {
	ipc_file_path = "ipc:///home/adhak001/dev/ipc_file";
	zmqContext = zmq_init(1);
	zmqRequester = zmq_socket(zmqContext, ZMQ_PUSH);
	int rc = zmq_connect(zmqRequester, ipc_file_path);
	assert (rc == 0);
	printf("ZMQ apparatus ready \n");
}

/* Function to send message to orchestrator */
int send_message_to_orchestrator(zmgr_msg * message) {
	//char buffer[6];
	size_t msg_size = message->msg_size;
	zmq_send(zmqRequester, message, msg_size, 0);
	//now wait for the reply
	//zmq_recv(zmqRequester, buffer, 6 ,0); // we only expect "OK" .. there is no need to process the message now
	rte_free(message);
	return 0;
}

/* creates a zmesg to be sent */
zmgr_msg *create_zmsg(pid_t pid[], int num_nfs,nf_state state) {
	zmgr_msg *new_msg = (zmgr_msg*) rte_malloc(NULL, sizeof(zmgr_msg), 0);
	new_msg->state = state;
	new_msg->msg_size = sizeof(zmgr_msg);

	if(num_nfs >1) {
		new_msg->information.num_of_elements = num_nfs;
		memcpy(new_msg->information.pid_array, pid, sizeof(pid_t)*num_nfs);
	}
	else
	{
		new_msg->information.num_of_elements = pid[0];
	}
	clock_gettime(CLOCK_MONOTONIC, &(new_msg->timestamp));
	return new_msg;
}

#endif //ONVM_GPU
