#include <stdio.h>
#include <cuda_runtime.h>
#include "onvm_stream.h"
#include <inttypes.h>
#include "onvm_common.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>//cuda error

//#define STREAM_PRIORITY_TEST

//Remove the below array.. and make an array that will rather store the pair of image data and nf_info pointers
struct gpu_callback gpu_callbacks[MAX_STREAMS * PARALLEL_EXECUTION];

stream_tracker streams_track[MAX_STREAMS];

/* initialize the streams  with no flags */
int init_streams(uint8_t priority) {
	int i;
	cudaError_t cuda_error;
	for (i = 0; i < MAX_STREAMS; i++) {
		if (!DEFAULT_STREAM) {
#ifdef STREAM_PRIORITY_TEST

			cuda_error = cudaStreamCreateWithPriority(&(streams_track[i].stream),
								cudaStreamNonBlocking, (i+1));
#else
			cuda_error = cudaStreamCreateWithFlags(&(streams_track[i].stream),
					cudaStreamNonBlocking);
#endif
			if (cuda_error != cudaSuccess) {
				printf("Failed to Create Streams. Priority was %"PRIu8" \n",priority);
				return -1;
			}
		} else {
			streams_track[0].stream = 0;
		}
		streams_track[i].status = PARALLEL_EXECUTION;
		streams_track[i].id = i;
		cudaEventCreate(&streams_track[i].event);
	}
	return 0;
}

/* return 0=all_released_and_free; 1=stream_still_busy_and_pending_to_complete */
int check_and_release_stream(void) {
	//check and return null of retry give_stream();
	int i;
	cudaError_t cuda_ret;
	for (i = 0; i < MAX_STREAMS; i++) {
		if (PARALLEL_EXECUTION > streams_track[i].status) {
			cuda_ret = cudaEventQuery(streams_track[i].event);
			if (cuda_ret == cudaSuccess) {
				//we can run callback here.
				gpu_image_callback_function(&streams_track[i].callback_info);
			} else {
				//	printf("CUDA error at give_stream_v2 error: %d \n",cuda_ret);
				return 1;
			}
		}
	}
	return 0;
}
//int status_tracker[MAX_STREAMS];
stream_tracker *give_stream_v2(void) {
	stream_tracker *st = give_stream();
	//int i = 0;
	//
	if (!st) {
		check_and_release_stream();
		st = give_stream();
	}
	return st;
}
int allowed_streams = MAX_STREAMS;
/* if the stream is available, then the stream will return otherwise it will return NULL, the client need to figure out what to do then */
stream_tracker *give_stream(void) {
	int i;
	int max = 0;
	int index;
	static long rr_counter = 0;
	if (PARALLEL_EXECUTION > 0) {
	//	for (i = 0; i < MAX_STREAMS; i++) { //changed to make dynamic stream provision
		for (i = 0; i < allowed_streams; i++) {
			if (streams_track[i].status > max) {
				max = streams_track[i].status;
				index = i;
			}
		}

		if (max) {
			streams_track[index].status--; //decrement
			// put in the timestamp
			clock_gettime(CLOCK_MONOTONIC, &streams_track[index].time_released);
			return &streams_track[index];
		} else {
			return NULL;
		}

		/*
		 if(streams_track[i].status>0)
		 {
		 streams_track[i].status--;
		 prev = i;
		 return &streams_track[i];
		 }
		 
		 } */
	} else {
		return &streams_track[(rr_counter++) % MAX_STREAMS];
	}
	return NULL;
}

/* new way of giving stream when there are multiple streams and you have to co-ordinate between them
 *
 */
stream_tracker *give_stream_v3(uint32_t observed_latency_us){
	//so we have to give stream such a way that if there are 2 streams, then, we only release the 2nd stream if the first stream's was released 90% of time of observed latency earlier
	int i = 0;
	int max = 0;
	//int index;
	static int last_used_index = 0;
	check_and_release_stream();
	if(!observed_latency_us){
		//first initial conditions, only give 1 stream at a time until we profile latency
		allowed_streams = 1;
		return give_stream_v2();
	}
	else
	{

		//now we have the latency value.. we should check if any stream is busy...
		allowed_streams = MAX_STREAMS;
		//get a timestamp.
		struct timespec current_time;
		clock_gettime(CLOCK_MONOTONIC, &current_time);

		//check if any stream are available
		for (i = 0; i < allowed_streams; i++) {
			if (streams_track[i].status > max) {
				max += streams_track[i].status;
			//	index = i;
			}
	}
		if(max>0){
		//index of another stream
			int another_stream = (1-last_used_index);
			//otherwise check if the timestamp is more than latency
			uint32_t time_diff = (current_time.tv_sec - streams_track[last_used_index].time_released.tv_sec)*1000000+(current_time.tv_nsec-streams_track[last_used_index].time_released.tv_nsec)/1000;
			//printf("Observed latency %"PRIu32" and time diff %"PRIu32"\n", observed_latency_us, time_diff);
			if(time_diff>=(0.75*observed_latency_us)){
				clock_gettime(CLOCK_MONOTONIC, &streams_track[another_stream].time_released);
				//printf("give stream\n");
				streams_track[another_stream].status--;
				last_used_index++;
				last_used_index = last_used_index%2;
				return &streams_track[another_stream];

				}
			else
				return NULL;
			}
	}
	//printf("All streams busy latency was %"PRIu32"\n", observed_latency_us);
	return NULL;
}

/* makes stream available for use again */
void return_stream(stream_tracker * stream) {
	stream->status++;
}

