#include <stdio.h>
#include "onvm_common.h"
#include "onvm_netml.h"
#include "onvm_gpu_buffer_factory.h"
#include "clipper_batchsize_extension.h"

#include <strings.h>//for ffs
#include <rte_mbuf.h>
#include <rte_ip.h>
#include <rte_udp.h>
#include <rte_byteorder.h>
#ifdef ONVM_GPU
//puts the data into 
#include "onvm_nflib.h"
//#define GPU_REALLOCATION_EXPERIMENT
#define NUM_OF_PKTS_PER_IMAGE 588

uint32_t data_aggregation_bulk_v2(void **pkts, unsigned nb_pkts, image_batched_aggregation_info_t *image_agg, void** drop_pkts, unsigned *db_pkts, __attribute((unused)) histogram_v2_t *arrival_latency) {
	//static placeholder variable for a single image
	void *payload;
	image_chunk_header_t *chunk_header;
	unsigned i = 0;
	struct rte_mbuf* pkt = NULL;
	uint32_t image_id = 0;

	//count all observed packets.
	static uint32_t number_of_packets_seen = 0;
	static uint8_t get_timestamp = 1;
	static struct timespec first_pkt_timestamp, last_pkt_timestamp;
#ifdef GPU_REALLOCATION_EXPERIMENT
	check_and_release_stream();
#endif //GPU_REALLOCATION_EXPERIMENT

	for (i = 0; i < nb_pkts; i++) {
		pkt = (struct rte_mbuf*) pkts[i];
		//we shall copy the packets here itself.. why should we give it to the handler
		if(onvm_pkt_ipv4_hdr(pkt) != NULL) {

			//update with per batch packet seen
			number_of_packets_seen += 1;

			//take the timestamp of the packet arrival.
			if(get_timestamp)
			{
				clock_gettime(CLOCK_MONOTONIC, &first_pkt_timestamp);
				get_timestamp = 0;
			}

			onvm_get_pkt_meta(pkt)->action = ONVM_NF_ACTION_DROP;
			//first find which image this packet belongs to
			payload = (void *)rte_pktmbuf_mtod_offset(pkt, void *, (sizeof(struct ether_hdr)+sizeof(struct ipv4_hdr)+sizeof(struct udp_hdr)));
			chunk_header =(image_chunk_header_t *)( (char * )payload + 2);// 2bytes offset to make it 4byte aligned address.
//printf("  ID %d start offset %"PRIu32" bytes length %"PRIu32" \n",  chunk_header->image_id, chunk_header->image_chunk.start_offset, chunk_header->image_chunk.size_in_bytes);
			image_id = (chunk_header->image_id);//*2;//TODO: hack to fix odd numbered image. DONE: Buffer overwriting incorrect index!


#ifdef NO_IMAGE_ID
			static uint current_image = 0;
			image_id = current_image;
			printf("Image ID to be referred to is %d \n",image_id);
#endif //NO_IMAGE_ID

			image_aggregation_info_t *image = &(image_agg->images[image_id]); //(image_agg->images[chunk_header->image_id]);


#ifdef NO_IMAGE_ID
		//There are cases where the fixed batch size cannot form fixed batch eg. final 5 images are filled but not 6th one for batch of 6
		// thus will never be able to update the image ID.. thus, we will just push the image ID by 1 if we encounter image with status 2.

		if(image->usage_status==2){

			current_image = (current_image+1)%MAX_IMAGES_BATCH_SIZE;
			image_id = current_image;
			image = &(image_agg->images[image_id]);
		}

#endif //NO_IMAGE ID

			if(image->usage_status==0 || image->bytes_count==0) {
				image->packets_count=0;
				image->usage_status = 1;
				clock_gettime(CLOCK_MONOTONIC, &image->first_packet_time);
			}

			if(1 == image->usage_status) {

				//now put the chunk in right place
				image->image_info.image_id = chunk_header->image_id;
				image->image_info.copy_info[image->packets_count].src_cpy_ptr = (void *)((char *)chunk_header+sizeof(image_chunk_header_t));
				image->image_info.copy_info[image->packets_count].image_chunk = chunk_header->image_chunk;

				//first put the rte_mbuf address in the proper place and update the packet counter
				image->image_packets[image->packets_count] = pkt;
				image->packets_count += 1;
				//now put the number of bytes
				image->bytes_count += chunk_header->image_chunk.size_in_bytes;

				//if we have a right amount of bytes for an image, we should make it a ready image and then update the readymask
				//if((image->packets_count == MAX_CHUNKS_PER_IMAGE)||(image->bytes_count >= SIZE_OF_AN_IMAGE_BYTES)) {
				if((image->bytes_count >= SIZE_OF_AN_IMAGE_BYTES)||(image->packets_count >= SIZE_OF_SENTENCE_BATCH))  //use for sentences NFs
				  {
					image->usage_status = 2;

					SET_BIT(image_agg->ready_mask,(image_id+1));
					clock_gettime(CLOCK_MONOTONIC, &image->last_packet_time);

					//time necessary to obtain an image
					//printf("Number of images in the batch right now %d \n",__builtin_popcountll(image_agg->ready_mask));

#ifdef NO_IMAGE_ID
					current_image = (current_image+1)%(MAX_IMAGES_BATCH_SIZE);
					//printf("Next image to be filled is %d.. \n", current_image);
#endif
				}
			} else {
				/*duplicate pkt for image that is already aggregated */
				onvm_get_pkt_meta(pkt)->action = ONVM_NF_ACTION_DROP;
				drop_pkts[(*db_pkts)++] = pkt;

				/* check if any GPU is free
				int k = 0;
				for(k = 0; k < NUM_GPUS; k++)
					check_and_release_stream(k);
					*/
			}

			//see if we have seen the number of packets that can make one image
			if(number_of_packets_seen>= NUM_OF_PKTS_PER_IMAGE){
				//we have a full image
				number_of_packets_seen -= NUM_OF_PKTS_PER_IMAGE;
				get_timestamp = 1; //reset the timestamp for next image
				clock_gettime(CLOCK_MONOTONIC, &last_pkt_timestamp);
				uint32_t time_in_ms = (last_pkt_timestamp.tv_sec-first_pkt_timestamp.tv_sec)*1000000+(last_pkt_timestamp.tv_nsec-first_pkt_timestamp.tv_nsec)/1000;
				hist_store_v2(arrival_latency,time_in_ms); //store the time in histogram
				//subtract the number
				//put the time taken for the image to gather in the histogram
				number_of_images_arrived_since_last_computation += 1;
			}


		} //end of if (ipv4 pkt)
		else {
			/*Non IPV4 Pkt: Just Drop */
			onvm_get_pkt_meta(pkt)->action = ONVM_NF_ACTION_DROP;
			drop_pkts[(*db_pkts)++] = pkt;
		}
	} //end of for

	//if(image_agg->ready_mask) {*ready_images_index=image_agg->ready_mask;} //if(ready_images) {*ready_images_index=ready_images;}

#ifdef HOLD_PACKETS_TILL_CALLBACK
	return 1;
#else
	return 0;
#endif
	return 0;
}

uint32_t data_aggregation(struct rte_mbuf *pkt, image_batched_aggregation_info_t *image_agg, uint32_t *ready_images_index) {
	uint32_t ready_images=0;
	//static placeholder variable for a single image
	void *payload;
	image_chunk_header_t *chunk_header;
	//first find which image this packet belongs to
	payload = (void *)rte_pktmbuf_mtod_offset(pkt, void *, (sizeof(struct ether_hdr)+sizeof(struct ipv4_hdr)+sizeof(struct udp_hdr)));

	chunk_header =(image_chunk_header_t *)( (char * )payload + 2);// 2bytes offset to make it 4byte aligned address.

	//printf("  ID %d start offset %"PRIu32" bytes length %"PRIu32" \n",  chunk_header->image_id, chunk_header->image_chunk.start_offset, chunk_header->image_chunk.size_in_bytes);

	uint32_t image_id = (chunk_header->image_id);//*2;//TODO: hack to fix odd numbered image. DONE: Buffer overwriting incorrect index!

#ifdef NO_IMAGE_ID
	static uint current_image = 0;
	image_id = current_image;
#endif

	image_aggregation_info_t *image = &(image_agg->images[image_id]); //(image_agg->images[chunk_header->image_id]);
	//image_aggregation_info_t *image2 = &(image_agg->images[image_id+1]);//(image_agg->images[chunk_header->image_id]);

	if(image->usage_status==0 || image->bytes_count==0) {
		image->packets_count=0;
		image->usage_status = 1;
		clock_gettime(CLOCK_MONOTONIC, &image->first_packet_time);
}

	if(1 == image->usage_status) {

		//now put the chunk in right place
		image->image_info.image_id = chunk_header->image_id;
		image->image_info.copy_info[image->packets_count].src_cpy_ptr = (void *)((char *)chunk_header+sizeof(image_chunk_header_t));
		image->image_info.copy_info[image->packets_count].image_chunk = chunk_header->image_chunk;

		//first put the rte_mbuf address in the proper place and update the packet counter
		image->image_packets[image->packets_count] = pkt;
		image->packets_count += 1;

		//now put the number of bytes
		image->bytes_count += chunk_header->image_chunk.size_in_bytes;

		//printf("chunk size bytes %d \n",chunk_header->image_chunk.size_in_bytes);

		//printf("packets count %d  ID %d start offset %"PRIu32" bytes length %"PRIu32" bytes_count %ld\n", image->packets_count, chunk_header->image_id, chunk_header->image_chunk.start_offset, chunk_header->image_chunk.size_in_bytes, image->bytes_count);
		//printf("Image chunk contents image id start offset %"PRIu32" and size is %"PRIu32"\n",image->copy_info.copy_info[image->packets_count].image_chunk.start_offset,image->copy_info.copy_info[image->packets_count].image_chunk.size_in_bytes);
		//printf("A value from packet %f \n",((float*)(image->copy_info.copy_info[image->packets_count].src_cpy_ptr))[0]);

		//if we have a right amount of bytes for an image, we should make it a ready image and then update the readymask
		//if((image->packets_count == MAX_CHUNKS_PER_IMAGE)||(image->bytes_count >= SIZE_OF_AN_IMAGE_BYTES)) {
		if((image->bytes_count >= SIZE_OF_AN_IMAGE_BYTES)) {
			image->usage_status = 2;
			//printf("Image %d is complete \n", image->image_info.image_id);
			SET_BIT(image_agg->ready_mask,(image_id+1));
			SET_BIT(ready_images, (image_id+1));
			clock_gettime(CLOCK_MONOTONIC, &image->last_packet_time);
			//++(*ready_images_count);
			//image_agg->ready_mask |= (1 << image_id);
			//image_agg->temp_mask |= (1<<image_id);
			//printf("Image mask : %"PRIu32"\n",image_agg->ready_mask);
			//image_id++;
			//image_id = (image_id%MAX_IMAGES_BATCH_SIZE);
#ifdef NO_IMAGE_ID
			current_image = (current_image+1)%MAX_IMAGES_BATCH_SIZE;

#endif
			//printf("The address of the bitmask %p \n", &ready_images);
		}
		//if(ready_images_index) (*ready_images_index) |= ready_images;
		if(ready_images) {*ready_images_index=ready_images;}
#ifdef HOLD_PACKETS_TILL_CALLBACK
		return 1;
#else
		return 0; //1;	//when 0 disable calbback release
#endif
	}
	//else {
	//what to do with this packet? < Duplicate for ready image cannot be handled.. so release it
	//*baggregated = 0;
	//return 0;
	//}
	//onvm_nflib_return_pkt(nf_info, pkt);
	//return ready_images;
	return 0;
}

/* the function to load and execute in GPU */
int load_data_to_gpu_and_execute(struct onvm_nf_info *nf_info,image_batched_aggregation_info_t * batch_agg_info, ml_framework_operations_t *ml_operations, cudaHostFn_t callback_function, uint64_t new_images) {
	int ret = 0;

	//printf("The bitmask %"PRIu32"\n",nf_info->image_info->ready_mask);
	__attribute__((unused)) static uint64_t last_processed_index = 0;//Note: need to use this to avoid starvation and not able to touch higher indexed imamges, when always overshooting.
//	__attribute__((unused)) static onvm_interval_timer_t start_tsc = 0;
//	__attribute__((unused)) static onvm_interval_timer_t end_tsc = 0;
//	__attribute__((unused)) static uint64_t busy_interval_tsc = 0;

	stream_tracker *cuda_stream = give_stream_v2();//give_stream();

	if(cuda_stream != NULL) {
	  	int gpu_id = cuda_stream->gpu_id;
		//printf("Load and execute got GPU number %d \n",gpu_id);
		cudaSetDevice(gpu_id);


		uint32_t i;

		//arguments for inference
		nflib_ml_fw_infer_params_t infer_params;

		//arguments for callback
		struct gpu_callback * callback_args = NULL;
		callback_args = &cuda_stream->callback_info;
#if 0  //This code is for explicit callback mode
		for(i =0; i<MAX_STREAMS*PARALLEL_EXECUTION; i++) {
			if(gpu_callbacks[i].status == 0) {
				gpu_callbacks[i].status = 1;
				callback_args = &(gpu_callbacks[i]);
				break;
			}
		}
		if(unlikely(NULL==callback_args)) {
			printf("Failed to get callback args\n");
			return_stream(cuda_stream);
			return 2;
		}
#endif //0

		//Check if images were remaining last time; then pick them.
		//if(unlikely(0 == last_processed_index)) {last_processed_index = new_images;}

		//for Freshness set (to avoid stale images comment below line
		//if(unlikely(nf_info->fixed_batch_size))
		last_processed_index|=new_images;

		uint32_t num_of_images = __builtin_popcountll(last_processed_index);//(last_processed_index)?(__builtin_popcount(last_processed_index)):(__builtin_popcount(new_images));
		//printf("Number of images we counted %"PRIu32" last processed index %"PRIu64" new images %"PRIu64"\n",num_of_images, last_processed_index,new_images);
		//num_of_images = (nf_info->fixed_batch_size)? ((num_of_images>nf_info->fixed_batch_size)?(nf_info->fixed_batch_size):(num_of_images)):(num_of_images);
		if(unlikely(nf_info->fixed_batch_size)) {
			if(num_of_images >= nf_info->fixed_batch_size) {
				num_of_images = nf_info->fixed_batch_size;
			} else {
				/* Not sufficient images in the current batch; hence wait till fixed_batch_Size is reached */
				//printf("Batch Size didn't reach\n");
				return_stream(cuda_stream);
				return 0;
			}
		}

		/* Should Adaptive batching be learning or not? */
		else if (ADAPTIVE_BATCHING_SELF_LEARNING == nf_info->enable_adaptive_batching) {
			// Check and cap to max batch size that is learnt and determined to not exceed SLO for the current operating settings
			if((nf_info->learned_max_batch_size) && (num_of_images > nf_info->learned_max_batch_size)) num_of_images = nf_info->learned_max_batch_size;
			//if((nf_info->learned_max_batch_size) && (num_of_images > nf_info->learned_max_batch_size)) num_of_images = nf_info->learned_max_batch_size;
			//adaptive batching help for getting beyond 32 images
			//if(num_of_images< nf_info->learned_max_batch_size){
			//	return_stream(cuda_stream);
			//	return 0;

			//}

		}



#ifdef CLIPPER_ADAPTIVE_BATCHING
		if(ADAPTIVE_BATCHING_SELF_LEARNING == nf_info->enable_adaptive_batching){
			int clipper_batch_size = clipper_check_batch_size(nf_info->inference_slo_ms*1000);
			//printf("Clipper predicted batch size: %d \n",clipper_batch_size);
			if(clipper_batch_size > MAX_IMAGES_BATCH_SIZE || clipper_batch_size <1 )
				clipper_batch_size = MAX_IMAGES_BATCH_SIZE;

			nf_info->learned_max_batch_size = clipper_batch_size;
			num_of_images = nf_info->learned_max_batch_size;
			//printf("Clipper: number of images %d\n",num_of_images);

			//if(num_of_images< nf_info->learned_max_batch_size){
			//	return_stream(cuda_stream);
			//	return 0;

		//	}
		}

#endif //clipper adaptive batching

		//(last_processed_index)?(last_processed_index):(new_images);
		uint64_t temp_bitmask = last_processed_index;//(last_processed_index)?(last_processed_index):(new_images);
		//uint32_t num_of_images = __builtin_popcount(new_images);

		void *in_buffers[MAX_IMAGES_PER_PARTITION] = {NULL,};
		void *out_buffers[MAX_IMAGES_PER_PARTITION] = {NULL,};
		void *in_cpu_buffers[MAX_IMAGES_PER_PARTITION] = {NULL,};
		void *out_cpu_buffers[MAX_IMAGES_PER_PARTITION] = {NULL,};
		uint32_t actual_images_in_batch = 0;
		uint64_t actual_images_in_batch_bitmask=0;
		void * input_dev_buffer = NULL;
		void * output_dev_buffer = NULL;
		void * cpu_side_buffer = NULL;
		void * cpu_side_output = NULL;

		for(i=0; i< num_of_images; i++) {
			//now get the GPU buffer for each image
			give_device_addresses(cuda_stream->id, &input_dev_buffer, &output_dev_buffer, gpu_id);
			//printf("GPU device input we got %p and device output buffer we got %p\n",input_dev_buffer,output_dev_buffer); 
			
			if(NULL == input_dev_buffer || NULL == output_dev_buffer) break;
			//last_processed_index=0;
			int index = ffsll(temp_bitmask);
			CLEAR_BIT(temp_bitmask, (index));
			SET_BIT(actual_images_in_batch_bitmask, (index));
			//SET_BIT(last_processed_index, index);
			actual_images_in_batch++;
			in_buffers[i] = input_dev_buffer;
			out_buffers[i] = output_dev_buffer;

			// for CPU buffer case
			give_cpu_addresses(cuda_stream->id,&cpu_side_buffer, &cpu_side_output);
			in_cpu_buffers[i] = cpu_side_buffer;
			out_cpu_buffers[i] = cpu_side_output;

		}
		//printf("number of images ready %d, can_be_processed=%d index of image %d \n", num_of_images, actual_images_in_batch, ffsll(new_images)-1);
		//printf("index of image being sent %d \n",ffsll(actual_images_in_batch_bitmask));
		//prepare execution arguments
		callback_args->bitmask_images= actual_images_in_batch_bitmask;//actual_images_in_batch;//new_images;
		callback_args->batch_aggregation = batch_agg_info;
		callback_args->stream_track = cuda_stream;
		callback_args->nf_info = nf_info;
		clock_gettime(CLOCK_MONOTONIC, &(callback_args->start_time));

		void * start_dev_buffer = in_buffers[0];
		void * start_output_buffer = out_buffers[0];

		//Compute statistics for the time to copy an image to GPU
		clock_gettime(CLOCK_MONOTONIC, &callback_args->start_gpu_transfer);


		//printf("Actual_images in batch %d\n",actual_images_in_batch);
		//printf("Image data structure %p Image index: ", nf_info->image_info);
		for(i = 0; i< actual_images_in_batch; i++) { //for(i = 0; i<num_of_images; i++) {
			//find which image is ready
			int image_index = ffsll(actual_images_in_batch_bitmask);// ffs(new_images);
			image_index -= 1;
			//printf("Image INDEX : %d ",image_index);
			//printf("images ready %d index %d \n",num_of_images, image_index);
			if(batch_agg_info->images[image_index].usage_status == 2) {

				//now get the GPU buffer for each image
				//give_device_addresses(cuda_stream->id, &input_dev_buffer, &output_dev_buffer);

				// for CPU buffer case
				//give_cpu_addresses(cuda_stream->id,&cpu_side_buffer, &cpu_side_output);

				//feed in the pointers of the packets to infrence args
				infer_params.array_of_packets[i] = (void **)batch_agg_info->images[image_index].image_info.copy_info;
				infer_params.num_packets[i]= batch_agg_info->images[image_index].packets_count;

				cpu_side_buffer = in_cpu_buffers[i];
				cpu_side_output = out_cpu_buffers[i];

				if(start_dev_buffer != NULL) {

					//change the status
					batch_agg_info->images[image_index].usage_status = 3;

					if(nf_info->gpu_percentage) {

#ifdef ENABLE_GPU_NETML
						//NetML transfer
						transfer_to_gpu((void *)(batch_agg_info->images[image_index].image_info.copy_info),batch_agg_info->images[image_index].packets_count,in_buffers[i],&(cuda_stream->stream));
#else
						//Copy Transfer
						if(nf_info->platform != pytorch)
							transfer_to_gpu_copy((void *)(batch_agg_info->images[image_index].image_info.copy_info),batch_agg_info->images[image_index].packets_count,cpu_side_buffer,in_buffers[i],&(cuda_stream->stream));
#endif

						CLEAR_BIT(actual_images_in_batch_bitmask, (image_index+1));	//CLEAR_BIT(new_images, (image_index+1));
						CLEAR_BIT(batch_agg_info->ready_mask, (image_index+1));
						CLEAR_BIT(last_processed_index, (image_index+1));
					} //checking GPU percentage
					  //printf("After posting image ready mask %"PRIu64",final_batch size %d \n", batch_agg_info->ready_mask, actual_images_in_batch);
				}
				else
				{
				  //batch_agg_info->images[image_index].usage_status = 0;
					printf("we could not get the GPU buffer\n");
					//break;

					//if we fail to secure a buffer, we need to clear the image and just proceed with next one
					//CLEAR_BIT(actual_images_in_batch_bitmask, (image_index+1));	//CLEAR_BIT(new_images, (image_index+1));
					//CLEAR_BIT(batch_agg_info->ready_mask, (image_index+1));
					//CLEAR_BIT(last_processed_index, (image_index+1));

					//return;
					break;

				}
			}
		}

		//timestamp for finished GPU transfer
		clock_gettime(CLOCK_MONOTONIC, &callback_args->end_gpu_transfer);

		//printf("\n");

		//time to execute the im`age
		//prepare execution arguments;

		//we have GPU available

		infer_params.batch_size = actual_images_in_batch;//__builtin_popcount(callback_args->bitmask_images);
		//printf("Batch size fed %d,\n",infer_params.batch_size);
		//printf("Batch size: %d Stream ID %"PRIu8" image mask %x\n",infer_params.batch_size, cuda_stream->id, callback_args->bitmask_images);
		infer_params.callback_data = callback_args;
		infer_params.callback_function = callback_function;
		infer_params.stream = &(cuda_stream->stream);
		infer_params.model_handle = nf_info->ml_model_handle[gpu_id];
		infer_params.gpu_id = gpu_id;

		//this path is different for CNTK and Tensorrt. CNTK only takes CPU side buffer not GPU side buffer so will only
		if(nf_info->gpu_model>5){
			infer_params.input_data = start_dev_buffer;
		}
		else if(nf_info->gpu_model<=5){
			infer_params.input_data = cpu_side_buffer;
		}
		infer_params.input_size = SIZE_OF_AN_IMAGE_BYTES*infer_params.batch_size;
		infer_params.output = start_output_buffer;

		//put in input and output pointers in callback args
		callback_args->input_data = infer_params.input_data;
		callback_args->output_data = infer_params.output;
		callback_args->input_size = infer_params.input_size;

		//struct timespec time_image_ready;
		//clock_gettime(CLOCK_MONOTONIC, &time_image_ready);
		//uint64_t image_ready_time = (time_image_ready.tv_sec)*1000000+(time_image_ready.tv_nsec)/1000;
		//printf("Image ready at time %"PRIu64"\n",image_ready_time);

		//conduct the inference.
		void * aio = NULL;
		//NO work in tensorrt now

		//struct timespec begin_infer, end_infer;
		//clock_gettime(CLOCK_MONOTONIC, &begin_infer);
		//printf("Before infering the images \n");
		int check_gpu;
		cudaGetDevice(&check_gpu);
		//printf("****----- Inferring the image in GPU %d ------ *** \n", check_gpu);
		ml_operations->infer_batch_fptr(&infer_params,aio );
		cudaEventRecord(cuda_stream->event,cuda_stream->stream);

		//printf("After calling TRT infer\n");
		//clock_gettime(CLOCK_MONOTONIC, &end_infer);

		//uint64_t infer_time = (end_infer.tv_sec-begin_infer.tv_sec)*1000000+(end_infer.tv_nsec-begin_infer.tv_nsec)/1000;
		//printf("Inference launch time (us) %"PRIu64" \n", infer_time);
	} else {
		//printf("GPU IS BUSY\n");
		return 1;// indicates busy
	}
	return ret;
}

/*
 void infer_the_image(uint32_t batch_size, void *callback_data, cudaHostFn_t callback_function,cudaStream_t *stream, void *model_handle, void* input_data, size_t input_size, float *output){

 nflib_ml_fw_infer_params_t infer_params;
 infer_params.batch_size = __builtin_popcount(callback_batch_info);
 infer_params.callback_data = callback_args;
 infer_params.callback_function = callback_function;
 infer_params.stream = &(cuda_stream->stream);
 infer_params.model_handle = nf_info->ml_model_handle;
 infer_params.input_data = start_dev_buffer;
 infer_params.input_size = SIZE_OF_AN_IMAGE_BYTES*infer_params.batch_size;
 infer_params.output = start_output_buffer;

 //conduct the inference.
 void * aio = NULL;
 //NO work in tensorrt now

 ml_operations->infer_batch_fptr(&infer_params,aio );

 return;
 }
 */

#endif
