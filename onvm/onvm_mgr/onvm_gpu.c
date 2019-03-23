#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <rte_malloc.h>

#include <zmq.h>
#include <assert.h>

#include "onvm_gpu.h"
#include "onvm_cntk_api.h"
#include "onvm_mgr.h"
#include "onvm_nf.h"

#ifdef ONVM_GPU
static inline struct onvm_nf_info *shadow_nf(int);
static inline struct onvm_nf_info *shadow_nf(int instance_id){
  //returns the info of shadow NF ID
  unsigned shadow_id = get_associated_active_or_standby_nf_id(instance_id);
  struct onvm_nf *cl;
  cl = &(nfs[shadow_id]);

  
  return cl->info;
}

void load_all_models(void){
  int i = 0;
  
  void * cpu_func_ptr = NULL;
  void * gpu_func_ptr = NULL;
  int flag = 1; //all models are loaded to GPU

  //initialize the file listing...
  file_listing = (struct gpu_file_listing*) rte_malloc(NULL,sizeof(struct gpu_file_listing)*NUMBER_OF_MODELS,0);
  
  
  for(i = 0; i<NUMBER_OF_MODELS; i++){
      /* convert the filename to wchar_t */
      size_t filename_length = strlen(ml_file_name[i]);
      wchar_t file_name_w[filename_length];
      mbstowcs(file_name_w,ml_file_name[i],filename_length+1);
 

      struct timespec begin,end;
      clock_gettime(CLOCK_MONOTONIC, &begin);
      load_model(file_name_w,&cpu_func_ptr, &gpu_func_ptr, flag, i);
      clock_gettime(CLOCK_MONOTONIC, &end);

      double time_taken_to_load = (end.tv_sec-begin.tv_sec)*1000000.0+(end.tv_nsec-begin.tv_nsec)/1000.0;
      printf("----- Time taken to load model %d is %f microseconds on GPU \n",i,time_taken_to_load);
      
      //now update the meta-data
      file_listing[i].file_index = i;
      memcpy(file_listing[i].model_name ,ml_model_name[i], strlen(ml_model_name[i]));
      file_listing[i].load_flag = flag;
      file_listing[i].cpu_function = cpu_func_ptr;
      file_listing[i].gpu_function = gpu_func_ptr;

      if(0){ //only available for vgg19 now.
	//update the time it takes to run the model.. read from the file.
	file_listing[i].attributes.run_times = rte_malloc(NULL, sizeof(float)*NUM_OF_RUNTIME_DATAPOINTS, 0);
	
	FILE *fp;
	char * line = NULL;
	size_t len = 0;
	ssize_t read;
	int counter = 0;
	
	fp = fopen(ml_model_name[i],"r");
	if (fp == NULL)
	  printf("Couldn't open %s\n", ml_model_name[i]);
	
	while((read = getline(&line, &len, fp)) != -1){
	  sscanf(line,"%f",&(file_listing[i].attributes.run_times[counter++])); //feed in the value
	}
      }

      //Now time to get file attributes..
      //general attributes..
      if(gpu_func_ptr != NULL)
	{
	  file_listing[i].attributes.number_of_parameters = count_parameters(gpu_func_ptr);
	  file_listing[i].attributes.number_of_inputs = count_inputs(gpu_func_ptr); //TODO

	  /* getting cuda handles */
	  cudaIpcMemHandle_t * mem_handles = (cudaIpcMemHandle_t *) rte_malloc(NULL, sizeof(cudaIpcMemHandle_t)*file_listing[i].attributes.number_of_parameters,0);
	  get_cuda_pointers_handles(gpu_func_ptr,mem_handles);
	  file_listing[i].attributes.cuda_handles = mem_handles;

	  /* todo, get the sizes of the input for the models */
	  int * inputs_sizes = (int *) rte_malloc(NULL,sizeof(int)*file_listing[i].attributes.number_of_inputs,0);
	  get_all_input_sizes(gpu_func_ptr,inputs_sizes,0); //please don't use this.. still working on it
	  file_listing[i].attributes.input_dimensions = inputs_sizes;

	}
      //update the attribute for CPU side.. 
      if(cpu_func_ptr != NULL)
	{
	  file_listing[i].attributes.number_of_parameters = count_parameters(cpu_func_ptr);
	  file_listing[i].attributes.number_of_inputs = count_inputs(cpu_func_ptr); //TODO
	  int * inputs_sizes = (int *) rte_malloc(NULL,sizeof(int)*file_listing[i].attributes.number_of_inputs,0);
	  get_all_input_sizes(cpu_func_ptr,inputs_sizes,0);
	  file_listing[i].attributes.input_dimensions = inputs_sizes;
	}
	
    }
  printf("Loaded all the ML files ..\n");
}

/* sends the GPU pointers for the NF */
void * provide_nf_with_model(struct onvm_nf_msg * msg){
  //resolve the message data
  //printf("---------- provide nf with model called \n");
  provide_gpu_model *query = (provide_gpu_model *) msg->msg_data;
  int index = query->model_index;

  //find which NF this came from.
  uint16_t nf_instance = query->nf->instance_id; //the instance ID of nf, this way we can get the msg key
  
  printf("--- sending ML model to NF instance %d ", nf_instance);
   
  //Now get all the attributes of the ML model we are checking
  return (void *) &(file_listing[index].attributes);
}

//check if the alternate is not null
  

void restart_nf(struct onvm_nf_info *nf){
  //get the PID and construct a zmsg and send it out to orchestartor
  printf("####____------#### restarting the NF %d \n", nf->pid);
  send_message_to_orchestrator(create_zmsg(&(nf->pid),1, zrestart));
}


void nf_is_gpu_ready(struct onvm_nf_info *nf){
  //TODO just print for now
  printf("NF instance %d is now ready to process packets \n", nf->instance_id);
}



/* ***********
 * The engine to compute if we need to change the percentage 
 ************
 * what do we need?
 * List of all the NFs that are using GPU.. and their Average Runtime of their program
 * The NFs should provide following things:
 * How many images they are getting Per_seconds
 * How many images they can process per_second
 * if the first one is greater than 2nd one.. we should consider resetting the percentage

 * We need to have this for multiple NFs and find optimal percentage
 * Already pre-compiled list of Runtime on different percentage for that NF
 */

void compute_GPU_allocation(struct onvm_nf_info *nf ){
  //get the reporting from an NF
  int nf_gpu_percentage = nf->gpu_percentage;
  float request_rate = nf->requests_per_second;
  float throughput = nf->images_throughput;
  float max_throughput = find_max_throughput(nf->gpu_model, nf_gpu_percentage);
  int recommended_gpu_percentage;
  
  printf(" nf_gpu_percentage %d request rate: %f and throughput %f \n",nf_gpu_percentage, request_rate, throughput);

  //recommend the NF to be restarted if the request per seconds are more than the throughput
  if(request_rate > 0.8*max_throughput)
    {
      // if the request rate is creeping up to 80% of max  we suggest that this NF be considered reallocating the resource
      nf->candidate_for_restart = 1;
      recommended_gpu_percentage = suggest_gpu_percentage(request_rate,nf->gpu_model);//find from the table

      //this means we are going to provide the shadow NF with the GPU percentage
      get_shadow_NF_ready(nf, recommended_gpu_percentage);

      //inform the NF that it is going to be restarted...
      inform_NF_of_pending_restart(nf);
    }
      
}

/* helper function to find the experimental throughput in for this model at this percentage */
float find_max_throughput(int model_index, int gpu_percentage){
  int i;
  int num_records = file_listing[model_index].attributes.num_of_runtimes;
  float *runtimes = file_listing[model_index].attributes.run_times;
  int *gpu_percentages = file_listing[model_index].attributes.gpu_percentages;
  for(i = 0; i<num_records; i++){
    if(gpu_percentages[i] >= gpu_percentage)
      return runtimes[i];
  }
  return runtimes[num_records-1];//send the best runtime
}

/* helper functions for model management */
int suggest_gpu_percentage(float request_rate, int model_index){
  //find the attributes...
  int num_records = file_listing[model_index].attributes.num_of_runtimes;
  float *runtimes = file_listing[model_index].attributes.run_times;
  int *gpu_percentages = file_listing[model_index].attributes.gpu_percentages;
  int i;
  for(i = 0; i < num_records; i++){
    if (runtimes[i]  >= 1.2*request_rate)
      return gpu_percentages[i];
  }
  return 100; //in case request rate is very large, give all the GPU to the program
}

/* the function name is self descriptive */
void inform_NF_of_pending_restart(struct onvm_nf_info *nf){
  onvm_nf_send_msg(nf->instance_id, MSG_RESTART, 0, NULL);
}

/* the function to send message to shadow NF */
void get_shadow_NF_ready(struct onvm_nf_info *shadow, int gpu_percentage){
  struct get_alternate_NF_ready* alternate_message = (void *)rte_malloc(NULL, sizeof(int)+sizeof(void*), 0);
  alternate_message->gpu_percentage = gpu_percentage;
  struct onvm_nf_info * alternate_nf = shadow_nf(shadow->instance_id);
  if(alternate_nf != NULL){
    alternate_message->image_info = alternate_nf->image_info;
    onvm_nf_send_msg((shadow_nf(shadow->instance_id))->instance_id, MSG_GET_GPU_READY, 0, alternate_message);
  }
}
/* ------- <MESASAGING API > ******** */
void init_zmq(void){
  ipc_file_path = "ipc:///home/adhak001/dev/ipc_file";
  zmqContext = zmq_init(1);
  zmqRequester  = zmq_socket(zmqContext, ZMQ_PUSH);
  int rc = zmq_connect(zmqRequester, ipc_file_path);
  assert (rc == 0);
  printf("ZMQ apparatus ready \n");
}


/* Function to send message to orchestrator */
int send_message_to_orchestrator(zmgr_msg * message){
  //char buffer[6];
  size_t msg_size = message->msg_size;
  zmq_send(zmqRequester, message, msg_size, 0);
  //now wait for the reply
  //zmq_recv(zmqRequester, buffer, 6 ,0); // we only expect "OK" .. there is no need to process the message now
  rte_free(message);
  return 0;
}

/* creates a zmesg to be sent */
zmgr_msg *create_zmsg(pid_t pid[], int num_nfs,nf_state state){
  zmgr_msg *new_msg = (zmgr_msg*) rte_malloc(NULL, sizeof(zmgr_msg), 0);
  new_msg->state = state;
  new_msg->msg_size = sizeof(zmgr_msg);

  if(num_nfs >1){
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
