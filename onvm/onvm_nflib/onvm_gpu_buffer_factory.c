#include <stdio.h>
#include "onvm_common.h"
#include "onvm_gpu_buffer_factory.h"
#ifdef ONVM_GPU
#include <cuda_runtime.h>

/* Function to resolve the cpu or GPU buffer */
void resolve_gpu_dev_buffer(void *input_ptr, void * output_ptr) {
#define OLD_GPU_ALLOCATION
#ifdef OLD_GPU_ALLOCATION
	cudaError_t cuda_return;
	cuda_return = cudaIpcOpenMemHandle(&input_dev_buffer,*((cudaIpcMemHandle_t *)input_ptr), cudaIpcMemLazyEnablePeerAccess);
	if(cuda_return != cudaSuccess) {
		printf("CUDA DEV IPC failed \n");
		printf("CUDA ERROR VALUE %d \n",cuda_return);
	}

	cuda_return = cudaIpcOpenMemHandle(&output_dev_buffer,*((cudaIpcMemHandle_t *)output_ptr),cudaIpcMemLazyEnablePeerAccess );

	if(cuda_return != cudaSuccess) {
		printf("CUDA DEV IPC failed \n");
	}
	/* also initialize the data structure which checks if the dev buffer is full or not
	 * initialize it to 0s meaning they are empty

	 * furthermore, compute the address of dev buffer by pointer arithmetic*/
	int i,j;
	for(i = 0; i < DEV_BUFFER_PARTITIONS; i++) {
		for(j = 0; j<MAX_IMAGES_PER_PARTITION;j++) {
			buffer_state[i].occupancy_indicator = 0;
			buffer_state[i].dev_buffers[j] = (char *)input_dev_buffer+(i*SIZE_OF_AN_IMAGE_BYTES); //this offsets the pointer with byte size to fit half the MAX_IMAGES_BATCH
			buffer_state[i].output_dev_buffers[j] = (char *)output_dev_buffer+(i*1000*sizeof(float));
		}
	}
#else
	printf("buffers from manager Input %p output %p\n",input_ptr, output_ptr);
	int i,j;
	void * cuda_memory;
	void *cuda_output;
	for(i = 0; i < DEV_BUFFER_PARTITIONS; i++) {
		cudaMalloc(&cuda_memory, (SIZE_OF_AN_IMAGE_BYTES*MAX_IMAGES_PER_PARTITION));
		cudaMalloc(&cuda_output, (MAX_IMAGES_PER_PARTITION*1000*4));
		for(j = 0; j<MAX_IMAGES_PER_PARTITION;j++) {
			buffer_state[i].occupancy_indicator = 0;
			buffer_state[i].dev_buffers[j] = (char *)cuda_memory+(i*SIZE_OF_AN_IMAGE_BYTES); //this offsets the pointer with byte size to fit half the MAX_IMAGES_BATCH
			buffer_state[i].output_dev_buffers[j] = (char *)cuda_output+(i*1000*sizeof(float));
		}
	}
#endif
	printf("Resolved GPU Dev Buffer \n");
}

/* similar function for CPU side buffer */
void resolve_cpu_side_buffer(void *input_buffer, void * output_buffer) {
	// furthermore, compute the address of dev buffer by pointer arithmetic
	input_cpu_buffer = input_buffer;
	output_cpu_buffer = output_buffer;
	int i,j;
	for(i = 0; i < DEV_BUFFER_PARTITIONS; i++) {
		for(j = 0; j<MAX_IMAGES_PER_PARTITION;j++) {
			cpu_buffer_state[i].occupancy_indicator = 0;
			cpu_buffer_state[i].cpu_buffers[j] = (char *)input_buffer+(i*SIZE_OF_AN_IMAGE_BYTES); //this offsets the pointer with byte size to fit half the MAX_IMAGES_BATCH
			cpu_buffer_state[i].output_cpu_buffers[j] = (char *)output_buffer+(i*1000*sizeof(float));
		}
	}
	printf("Resolved CPU Dev Buffer \n");
}

/* the function that provides the GPU address */
void give_cpu_addresses(uint8_t batch_id, void ** input_buffer, void ** output_buffer) {
	/* check through the buffers to see which one are empty and provide.. otherwise send NULL*/

	//printf("Device address called \n");
	if(cpu_buffer_state[batch_id].occupancy_indicator < MAX_IMAGES_PER_PARTITION ) {
		//good, we have found a buffer which already belongs to this batch and have some space left.
		*input_buffer = cpu_buffer_state[batch_id].cpu_buffers[cpu_buffer_state[batch_id].occupancy_indicator];
		*output_buffer = cpu_buffer_state[batch_id].output_cpu_buffers[cpu_buffer_state[batch_id].occupancy_indicator++];
	}
	else {
		//unfortunately either there is no buffer for this batch or the buffer is full. so we need to give address from empty buffer
		*input_buffer = NULL;
		*output_buffer = NULL;
	}
}

/* the function that makes available the device buffer after it is used.. typically called by the callback function */
void return_cpu_buffer(uint8_t batch_id) {

	if(batch_id<DEV_BUFFER_PARTITIONS) {
		cpu_buffer_state[batch_id].occupancy_indicator = 0;
		return;
	}
	else
	printf("ERROR! COULDN'T FIND THE DEVICE BUFFER FOR BATCH %d \n", batch_id);
	return;
}

void resolve_gpu_dev_buffer_pointer(void *input_dev_buffer, void *output_dev_buffer) {

	/* also initialize the data structure which checks if the dev buffer is full or not
	 * initialize it to 0s meaning they are empty

	 * furthermore, compute the address of dev buffer by pointer arithmetic*/
	int i,j;
	for(i = 0; i < DEV_BUFFER_PARTITIONS; i++) {
		for(j = 0; j<MAX_IMAGES_PER_PARTITION;j++) {
			buffer_state[i].occupancy_indicator = 0;
			buffer_state[i].dev_buffers[j] = (char *)input_dev_buffer+(i*SIZE_OF_AN_IMAGE_BYTES); //this offsets the pointer with byte size to fit half the MAX_IMAGES_BATCH
			buffer_state[i].output_dev_buffers[j] = (char *)output_dev_buffer+(i*1000*sizeof(float));
		}
	}
	printf("Resolved GPU Dev Buffer \n");

}

/* the function that provides the GPU address */
void give_device_addresses(uint8_t batch_id, void ** input_buffer, void ** output_buffer) {
	/* check through the buffers to see which one are empty and provide.. otherwise send NULL*/

	//printf("Device address called \n");
	if(buffer_state[batch_id].occupancy_indicator < MAX_IMAGES_PER_PARTITION ) {
		//good, we have found a buffer which already belongs to this batch and have some space left.
		*input_buffer = buffer_state[batch_id].dev_buffers[buffer_state[batch_id].occupancy_indicator];
		*output_buffer = buffer_state[batch_id].output_dev_buffers[buffer_state[batch_id].occupancy_indicator++];
	}
	else {
		//unfortunately either there is no buffer for this batch or the buffer is full. so we need to give address from empty buffer
		*input_buffer = NULL;
		*output_buffer = NULL;

		//ADITYA's MOD
		//TODO: REmove this part as this part gives some buffer to program all the time.
		*input_buffer = buffer_state[batch_id].dev_buffers[0];
		*output_buffer = buffer_state[batch_id].dev_buffers[0];
	}
}

/* the function that makes available the device buffer after it is used.. typically called by the callback function */
void return_device_buffer(uint8_t batch_id) {

	if(batch_id<DEV_BUFFER_PARTITIONS) {
		buffer_state[batch_id].occupancy_indicator = 0;
		return;
	}
	else
	printf("ERROR! COULDN'T FIND THE DEVICE BUFFER FOR BATCH %d \n", batch_id);
	return;
}

#endif
