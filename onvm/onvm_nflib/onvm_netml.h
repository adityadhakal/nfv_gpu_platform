#ifndef _ONVM_NETML_H
#define _ONVM_NETML_H

//#include "onvm_common.h"

//#ifdef ONVM_GPU
#include "onvm_ml_libraries.h"
#include "onvm_stream.h"
#include "histogram.h"

#define ENABLE_GPU_NETML
//#define NO_IMAGE_ID //enables image packets to be places without caring about which file they belong to


#define MAX_CHUNKS_PER_IMAGE 4332
#define MAX_IMAGES_BATCH_SIZE 64

#define SIZE_OF_EACH_ELEMENT sizeof(float)

#define SIZE_OF_AN_IMAGE_BYTES (SIZE_OF_EACH_ELEMENT*3*416*416)
#define IMAGE_BATCH_DEV_BUFFER_SIZE (MAX_IMAGES_BATCH_SIZE*SIZE_OF_AN_IMAGE_BYTES)

#define SIZE_OF_SENTENCE_BATCH 10000 //set this to large value for images execution

/* structure that defines a chunk of data included in a packet*/
typedef struct __attribute__ ((packed)) chunk_info_t {
	uint32_t start_offset;
	uint32_t size_in_bytes;
} chunk_info_t;

/* struct that defines the image chunk header */
typedef struct __attribute__ ((packed)) image_chunk_header_t {
	//char padding1;
	//char padding2;
	uint32_t image_id;
	chunk_info_t image_chunk;
} image_chunk_header_t;

/* struct that points to the raw data that goes to GPU*/
typedef struct chunk_copy_info_t {
	void *src_cpy_ptr;
	chunk_info_t image_chunk;
} chunk_copy_info_t;

/* struct that tracks the single image's chunks */
typedef struct image_copy_info_t {
	uint32_t image_id;
	chunk_copy_info_t copy_info[MAX_CHUNKS_PER_IMAGE];
} image_copy_info_t;

/* struct that defines the status of each aggregated image */
typedef struct image_aggregation_info_t {
	uint8_t usage_status; // 0-free, 1-aggregating, 2-ready, 3-sent_to_inference, 4-inference_complete, 3-sent_to_copy, 4-copy_complete 
	size_t bytes_count;
	uint16_t packets_count;
	struct timespec first_packet_time;
	struct timespec last_packet_time;
	image_copy_info_t image_info;
	struct rte_mbuf * image_packets[MAX_CHUNKS_PER_IMAGE];
} image_aggregation_info_t;

/* the struct that NF really accesses */
typedef struct image_batched_aggregation_info_t {
	uint64_t ready_mask;
	image_aggregation_info_t images[MAX_IMAGES_BATCH_SIZE];
	//additional info for counting number of images
	struct timespec first_execution;
	uint32_t num_of_requests_inferred;
} image_batched_aggregation_info_t;

inline int get_recent_ts(struct timespec smaller, struct timespec bigger) {
	if (smaller.tv_sec < bigger.tv_sec)
		return 0; //bigger is the recent one
	else if ((smaller.tv_sec == bigger.tv_sec)
			&& (smaller.tv_nsec < bigger.tv_nsec))
		return 0; //bigger (Second) is the recent.
	return 1; //smaller (first) is the recent
}

struct stream_tracker;
//callback struct, for GPU callback
typedef struct gpu_callback {
	struct onvm_nf_info *nf_info;
	uint8_t status; // 0- available 1-in use
	image_batched_aggregation_info_t *batch_aggregation;
	uint64_t bitmask_images;
	struct timespec start_time;
	struct timespec start_gpu_transfer;
	struct timespec end_gpu_transfer;
	struct stream_tracker *stream_track;
} gpu_callback;

#define MAX_STREAMS 1
#define PARALLEL_EXECUTION 1
#define STREAMS_ENABLED 1
#define DEFAULT_STREAM 1
struct gpu_callback;

typedef struct stream_tracker {
	cudaStream_t stream;
	uint8_t status; //0 - being used in 2 executions, 1 being used in 1 execution, 1 slot available, 0 - available
	uint8_t id;	  //0 - id of the stream
	cudaEvent_t event;
	gpu_callback callback_info; //information for callback.
	struct timespec time_released; //timestamp of last "give_stream"
} stream_tracker;

extern struct gpu_callback gpu_callbacks[MAX_STREAMS * PARALLEL_EXECUTION];

/* Callback function after GPU process has ended*/
void gpu_image_callback_function(void *data);

/* this function initializes the number of streams desired by the program */
// providing functionality for stream priority
int init_streams(uint8_t priority);

/* this function provides an empty stream */
stream_tracker *give_stream_v3(uint32_t observed_latency_us);
stream_tracker *give_stream_v2(void);
stream_tracker *give_stream(void);
int check_and_release_stream(void);
/* this function returns stream */
void return_stream(stream_tracker *stream);

/* Functions */
// void as there will be a data transfer callback from GPU that will update the stats, there is nothing this function needs to return
// Return Status: 1 indicates packet is enqueued and no need to process further, 0 indicates rejected, so drop/fwd/return the packet.
uint32_t data_aggregation(struct rte_mbuf *pkt,
		image_batched_aggregation_info_t *image_aggregation_info,
		uint32_t *ready_images_index);
uint32_t data_aggregation_bulk_v2(void **pkts, unsigned nb_pkts,
		image_batched_aggregation_info_t *image_agg, void** drop_pkts,
		unsigned *db_pkts, histogram_v2_t *arrival_histogram);

void transfer_to_gpu(void *data_ptrs, int number_of_data_pts, void *destination,
		cudaStream_t *stream);

void transfer_to_gpu_copy(void * data_ptrs, int num_of_payload_data,
		void *cpu_destination, void * gpu_destination, cudaStream_t *stream);

/* the function to load and execute in GPU */
int load_data_to_gpu_and_execute(struct onvm_nf_info *nf_info,
		image_batched_aggregation_info_t * batch_agg_info,
		ml_framework_operations_t *ml_operations,
		cudaHostFn_t callback_function, uint64_t new_images);

void check_kernel(void *ptr);
#endif
//#endif
