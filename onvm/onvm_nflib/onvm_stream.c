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
/* if the stream is available, then the stream will return otherwise it will return NULL, the client need to figure out what to do then */
stream_tracker *give_stream(void) {
	int i;
	int max = 0;
	int index;
	static long rr_counter = 0;
	if (PARALLEL_EXECUTION > 0) {
		for (i = 0; i < MAX_STREAMS; i++) {
			if (streams_track[i].status > max) {
				max = streams_track[i].status;
				index = i;
			}
		}

		if (max) {
			streams_track[index].status--; //decrement
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

/* makes stream available for use again */
void return_stream(stream_tracker * stream) {
	stream->status++;
}

