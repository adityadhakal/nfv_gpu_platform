#ifndef ONVM_GPU_H
#define ONVM_GPU_H

#include <string.h>
#include <time.h>

#include "onvm_common.h"
#include "onvm_cntk_api.h"



/* a function to load all the existing models */
void load_all_models(void);

/* file name business */
char ml_file_name[NUMBER_OF_MODELS][100];
const char * ml_model_name[NUMBER_OF_MODELS];

#define NUM_OF_RUNTIME_DATAPOINTS 10

//variable for storing all models information //this struct is in onvm_common.h
struct gpu_file_listing * file_listing;

/* actually fill in the file paths */
const char *model_dir;


/* loading the filename.... now we will have fixed number of files hardcoded */
static inline void set_filename(void){
  model_dir = "/home/adhak001/openNetVM-dev/ml_models/";
  strcpy(ml_file_name[0], model_dir);
  strcat(ml_file_name[0],"AlexNet_ImageNet_CNTK.model");
  ml_model_name[0] = "alexnet_runtime.txt";
  strcpy(ml_file_name[1], model_dir);
  strcat(ml_file_name[1],"ResNet50_ImageNet_CNTK.model");
  ml_model_name[1] = "resnet50_runtime.txt";
  strcpy(ml_file_name[2], model_dir);
  strcat(ml_file_name[2],"VGG19_ImageNet_Caffe.model");
  ml_model_name[2] = "vgg19_rutime.txt";
  strcpy(ml_file_name[3], model_dir);
  strcat(ml_file_name[3],"ResNet152_ImageNet_CNTK.model");
  ml_model_name[3] = "resnet152_runtime.txt";
  strcpy(ml_file_name[4], model_dir);
  strcat(ml_file_name[4],"Fast-RCNN_Pascal.model");
  ml_model_name[4] = "Fast-RCNN_runtime.txt";
  strcpy(ml_file_name[5], model_dir);
  strcat(ml_file_name[5],"SRGAN.model");
  ml_model_name[5] = "SRGAN_runtime.txt";  
}



/* a function that provides NF with model's GPU pointers */
void * provide_nf_with_model(struct onvm_nf_msg *msg);

/*we can restart the NF safely now */
void restart_nf(struct onvm_nf_info *nf);

/* inform the NF should get ready for restart */
void inform_NF_of_pending_restart(struct onvm_nf_info *nf);

/*get the shadow NF ready */
void get_shadow_NF_ready(struct onvm_nf_info *nf, int recommended_gpu_percentage);

/* we know the shadow NF is ready for GPU execution, can restart the original NF if it is restart ready */
void nf_is_gpu_ready(struct onvm_nf_info *nf);

//these two functions should check the nf_info struct so they only send the restart once.

/* NF says it is okay to be restarted... restart it only if the shadow NF is ready */
void nf_is_okay_to_restart(struct onvm_nf_info *nf);

/* Helper functions for finding the GPU percent wise compute time ... computed by undergrads */
int suggest_gpu_percentage(float request_rate, int gpu_model);

/* find the throughput for certain model at certain percentage */
float find_max_throughput(int gpu_model, int gpu_percentage);

/* the model that computes the GPU allocation and then recommends the new GPU percentage */
void compute_GPU_allocation(struct onvm_nf_info *nf);

#ifdef ONVM_GPU_TEST
/* the function to test above apparatus */
void voluntary_restart_the_nf(struct onvm_nf_info *nf);
#endif

/*
static inline struct onvm_nf_info *shadow_nf(int instance_id){
  if(instance_id > 8)
    instance_id -= 8;
  else
    instance_id += 8;

  struct onvm_nf *cl;
  cl = &(nfs[instance_id]);
  
  return cl->info;
}
*/
/* Data for ZMQ message passing */

/* Enum for orchestrator */
typedef enum nf_state{zstart, zrestart, zstop} nf_state;

typedef struct zinformation_format{
  int num_of_elements;
  int pid_array[5];
}zinfo_format;

/* message passing struct between ONVM manager and orchestrator */
typedef struct zmgr_msg_struct{
  nf_state state;
  size_t msg_size;
  struct timespec timestamp;
  zinfo_format information; //size of this information have to be known... I recommend an int (4 bytes) + int array of size 5 (20 bytes)
} zmgr_msg;
//check the information format below


//some variables
void * zmqContext;
void * zmqRequester;
const char * ipc_file_path;// = "ipc:///home/adhak001/dev/ipc_file";

/* the function to init the zmq */
void init_zmq(void);

/* the function to send the message to orchestrator */
int send_message_to_orchestrator(zmgr_msg *message);

zmgr_msg * create_zmsg(pid_t nf_pid[], int num_of_nfs, nf_state state);
#endif
