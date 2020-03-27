/*********************************************************************
 *                     openNetVM
 *       https://github.com/sdnfv/openNetVM
 *
 *  Copyright 2015 George Washington University
 *            2015 University of California Riverside
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 * monitor.c - an example using onvm. Print a message each p package received
 ********************************************************************/

#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdarg.h>
#include <errno.h>
#include <sys/queue.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>

#include <rte_common.h>
#include <rte_mbuf.h>
#include <rte_ip.h>
#include <rte_atomic.h>
#include <rte_memzone.h>

#include "onvm_nflib.h"
#include "onvm_pkt_helper.h"
#include "onvm_netml.h"
#include "ml_load_balancer.h"

#define NF_TAG "ml_load_balancer"



//#define FAKE_COMPUTE
#define FACT_VALUE 30
long factorial(int n);
long factorial(int n)
{
        long result;
        if (n == 0 || n == 1) {
                result = 1;
        }
        else {
                result = factorial(n-1) * n;
        }

        return result;
}

/* Struct that contains information about this NF */
struct onvm_nf_info *nf_info;
int data_aggregation_lb(uint32_t image_id,image_batched_aggregation_info_t *image_agg,image_chunk_header_t *chunk_header, struct rte_mbuf* pkt, uint8_t dest_nf_id);

//where should be packet be placed
static inline int pkt_to_request(image_batched_aggregation_info_t *image_agg, uint32_t image_id);


/* number of package between each print */
static uint32_t print_delay = 1000000;
static uint32_t destination = 0;
static uint16_t dst_flag = 0;

// Address of data structure of IFs
void *if_data[MAX_NFS/2];
struct onvm_nf *nfs;

// the service we will be load balancing for
int preffered_service = 3;
/*
 * Print a usage message
 */
static void
usage(const char *progname) {
        printf("Usage: %s [EAL args] -- [NF_LIB args] -- -p <print_delay>\n\n", progname);
}

/*
 * Parse the application arguments.
 */
static int
parse_app_args(int argc, char *argv[], const char *progname) {
        int c;
        printf("Processing Args!!!!!!!!!!!1");
        while ((c = getopt (argc, argv, "d:p:")) != -1) {
                switch (c) {
                case 'p':
                        print_delay = strtoul(optarg, NULL, 10);
                        RTE_LOG(INFO, APP, "print_delay = %d\n", print_delay);
                        break;
                case 'd':
                        destination = strtoul(optarg, NULL, 10);
                        if (destination) dst_flag = 1;
                        printf("Destination_flag: %d, Destination: %d", dst_flag, destination); 
                        break;
                case '?':
                        usage(progname);
                        if (optopt == 'p')
                                RTE_LOG(INFO, APP, "Option -%c requires an argument.\n", optopt);
                        else if (isprint(optopt))
                                RTE_LOG(INFO, APP, "Unknown option `-%c'.\n", optopt);
                        else
                                RTE_LOG(INFO, APP, "Unknown option character `\\x%x'.\n", optopt);
                        return -1;
                default:
                        usage(progname);
                        return -1;
                }
        }
        return optind;
}

static void
do_additional_stat_display(void) {
        static uint64_t last_cycles;
        static uint64_t cur_pkts = 0;
        static uint64_t last_pkts = 0;
        uint64_t cur_rate = 0;
        static uint64_t peak_rate = 0;
        static uint64_t min_rate  = 0;

        uint64_t cur_cycles = rte_get_tsc_cycles();
        cur_pkts += print_delay;

        cur_rate =  ((cur_pkts - last_pkts) * rte_get_timer_hz()) / (cur_cycles - last_cycles);
        peak_rate = (cur_rate > peak_rate)? (cur_rate):(peak_rate);
        min_rate = (min_rate == 0 || min_rate > cur_rate)? (cur_rate):(min_rate);        

        printf("Total packets: %9"PRIu64" \n", cur_pkts);
        printf("TX pkts per second: %9"PRIu64" \n", cur_rate);
        printf("TX pkts Max rate: %9"PRIu64" and  Min rate: %9"PRIu64" \n", peak_rate, min_rate);

        last_pkts = cur_pkts;
        last_cycles = cur_cycles;

        printf("\n\n");
}
/*
 * This function displays stats. It uses ANSI terminal codes to clear
 * screen when called. It is called from a single non-master
 * thread in the server process, when the process is run with more
 * than one lcore enabled.
 */
__attribute__((unused)) static void
do_stats_display(struct rte_mbuf* pkt) {
        const char clr[] = { 27, '[', '2', 'J', '\0' };
        const char topLeft[] = { 27, '[', '1', ';', '1', 'H', '\0' };
        static int pkt_process = 0;
        struct ipv4_hdr* ip;

        pkt_process += print_delay;

        /* Clear screen and move to top left */
        printf("%s%s", clr, topLeft);

        printf("PACKETS\n");
        printf("-----\n");
        printf("Port : %d\n", pkt->port);
        printf("Size : %d\n", pkt->pkt_len);
        printf("Hash : %u\n", pkt->hash.rss);
        printf("N°   : %d\n", pkt_process);
        printf("\n\n");

        ip = onvm_pkt_ipv4_hdr(pkt);
        if (ip != NULL) {
                onvm_pkt_print(pkt);
        } else {
                printf("No IP4 header found\n");
        }
        do_additional_stat_display();
}

/* functions to wake up the client */
static inline void
notify_client_if(__attribute__((unused)) int instance_id)
{
	static sem_t *mutex;
	mutex = all_mutex[instance_id];
#ifdef USE_SEMAPHORE
        sem_post(mutex);
#endif
}

/* check if IF is sleeping or not and wake it up if it is sleeping */
static inline void check_and_wakeup_if(uint16_t instance_id) {

	rte_atomic16_t *shm = shm_server[instance_id];

	if (rte_atomic16_read(shm) ==1) {
			rte_atomic16_set(shm, 0);
			notify_client_if(instance_id);
#ifdef ENABLE_NF_WAKE_NOTIFICATION_COUNTER
			nfs[instance_id].stats.wakeup_count+=1;
#endif
        }
}

int data_aggregation_lb(uint32_t image_id,image_batched_aggregation_info_t *image_agg,image_chunk_header_t *chunk_header, struct rte_mbuf* pkt, uint8_t dest_nf_id);
int data_aggregation_lb(uint32_t image_id,image_batched_aggregation_info_t *image_agg,image_chunk_header_t *chunk_header, struct rte_mbuf* pkt, uint8_t dest_nf_id){
#ifdef NO_IMAGE_ID
			static uint current_image = 0;
			image_id = current_image;
			//printf("Image ID to be referred to is %d \n",image_id);
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
					//printf("making IF %d infer the image %d.\n",dest_nf_id,chunk_header->image_id);

					SET_BIT(image_agg->ready_mask,(image_id+1));
					clock_gettime(CLOCK_MONOTONIC, &image->last_packet_time);

					committed_request[chunk_header->image_id] = 0;
					//printf("waking up the IF %"PRIu8"\n",dest_nf_id);
					//cleanup the mapping between image ID and buffer ID
					image_agg->image_to_buffer_mapping[image->image_info.image_id] = 0;

					//Make destination NF check that it has an image pending
					check_and_wakeup_if(dest_nf_id);
#ifdef NO_IMAGE_ID
					current_image = (current_image+1)%(MAX_IMAGES_BATCH_SIZE);
#endif
				}
				return 0;
			}

			//Make destination NF check that it has an image pending
			//check_and_wakeup_if(dest_nf_id);
			return 1; //return if we could not put the packet in there.
}




static int
packet_handler(struct rte_mbuf* pkt, __attribute__((unused)) struct onvm_pkt_meta* meta, __attribute__((unused))  struct onvm_nf_info *nf_info) {

	/* packet management, drop it so the TX ring in end can drop it */
	meta->action=ONVM_NF_ACTION_DROP;
	meta->destination = pkt->port;
	/* check the packet to find out which file it is */
	//static placeholder variable for a single image
	void *payload;
	image_chunk_header_t *chunk_header;
	//unsigned i = 0;
	uint32_t image_id = 0;
	int retval = 0;

	struct ipv4_hdr* ip;

    ip = onvm_pkt_ipv4_hdr(pkt);
    if (ip != NULL) {

	//first find which image this packet belongs to
	payload = (void *)rte_pktmbuf_mtod_offset(pkt, void *, (sizeof(struct ether_hdr)+sizeof(struct ipv4_hdr)+sizeof(struct udp_hdr)));
	chunk_header =(image_chunk_header_t *)( (char * )payload + 2);// 2bytes offset to make it 4byte aligned address.

	//image_id = (chunk_header->image_id);

	//find the mapping of image ID to NF instance
	uint8_t nf_instance = committed_request[chunk_header->image_id];

	//if this request is not committed yet
	if(nf_instance == 0){
		//get which nf_instance this request has to be directed
		nf_instance = pkt_to_if(chunk_header->image_id);
		if(nf_instance == nf_info->instance_id){
			printf("No where to place the packet\n");
			return 1;
		}
	}

	//check where this image can be placed in the data structure;
	image_batched_aggregation_info_t *image_agg = (image_batched_aggregation_info_t *) if_data[nf_instance];

	//Image ID and buffer ID are decoupled
	image_id = pkt_to_request(image_agg,chunk_header->image_id); //NOTE: THIS VARIABLE NAME SHOULD BE BUFFER ID
	//printf("Image_ID %"PRIu32"\n",image_id);

	//put the image in the right data structure
	retval =  data_aggregation_lb(image_id, image_agg,chunk_header,pkt, nf_instance);
	/*if(unlikely(retval==1))
	{
		rte_pktmbuf_free(pkt);
	}
	*/
    }
	return retval;

}

uint8_t pkt_to_if(uint32_t image_id){
	// commit this image to IF
	//let's just round robin this for now

	//if we have no services, return the instance ID of loadbalancer, so the packet can be dropped
	if(services_count[preffered_service] == 0)
		return nf_info->instance_id;

	static int counter = 0;
	uint16_t num_services =  services_count[preffered_service];

	//this is where we check the each instance's stats to make decision on where to commit this image
	committed_request[image_id] = registered_services[(counter++)% num_services];
	//printf("Committed to the %"PRIu8"\n",committed_request[image_id]);
	return committed_request[image_id];

	//check all the services and their
}

//where should be packet be placed
static inline int pkt_to_request(image_batched_aggregation_info_t *image_agg, uint32_t image_id){
	//if image/request already has a buffer return it
	if(image_agg->image_to_buffer_mapping[image_id]>0)
	{
		//printf("buffer ID: %"PRIu8"\n",image_agg->image_to_buffer_mapping[image_id]-1);
		return (image_agg->image_to_buffer_mapping[image_id]-1);
	}
	else
	{
		//if the image/request doesn't have a buffer commit it to one
		int i = 0;
		for(i = 0; i<MAX_IMAGES_BATCH_SIZE; i++){
			if(image_agg->images[i].usage_status==0){
				//commit the new image to that buffer ID
				image_agg->image_to_buffer_mapping[image_id] = i+1;
				//printf("buffer Returned from loop: %d\n",i);
				return i;
			}

		}

	}
	//if there is no buffer left... we have a full queue
	return nf_info->instance_id;

}

static inline void attach_to_shm(uint16_t instance_id) {

	int shmid;
	char * shm;
	key_t key;
	key = get_rx_shmkey(instance_id);

	if ((shmid = shmget(key, SHMSZ, 0666)) < 0) {
		perror("shmget");
		fprintf(stderr, "unable to Locate the segment for client %d\n", instance_id);
		exit(1);
	}

	if ((shm = shmat(shmid, NULL, 0)) == (char *) -1) {
		fprintf(stderr, "can not attach the shared segment to the client space for client %d\n", instance_id);
		exit(1);
	}

	shm_server[instance_id] = (rte_atomic16_t *)shm;
}

static inline void open_mutex(uint16_t instance_id){
		const char *sem_name;
		sem_name = get_sem_name(instance_id);

		fprintf(stderr, "sem_name=%s for client %d\n", sem_name, instance_id);

		all_mutex[instance_id] = sem_open(sem_name, 0, 0666, 0);
			if (all_mutex[instance_id] == SEM_FAILED) {
				perror("Unable to execute semaphore");
				fprintf(stderr, "unable to execute semphore for client %d\n", instance_id);
				sem_close(all_mutex[instance_id]);
				exit(1);
			}
}

void initialize_if_mempools(void);
/* function to store the address of the mempools of the inference functions */
void initialize_if_mempools(void){
	struct rte_mempool * image_batch_aggregation_info;
	int i = 0;
	int retval;
	for(i = 0; i < (MAX_NFS/2); i++){
		// Get the rte_mempool and put it back
		image_batch_aggregation_info = rte_mempool_lookup(get_image_batch_agg_name(i));
		retval = rte_mempool_get(image_batch_aggregation_info,(void **)&(if_data[i]));
		printf("return value of mempool get: %d for IF %d\n", retval,i);
		//Return the image buffer
		rte_mempool_put(image_batch_aggregation_info,if_data[i]);
		//check if any IF is running
		if((&nfs[i])->info && ((&nfs[i])->info->status & NF_RUNNING)){
				printf("NF: %d is valid\n",i);

				//if the NF is VALID, we have to attached to the shared memory for the Mutex
				attach_to_shm(i);

				//open MUTEX
				open_mutex(i);
		}
	}
	// For test: all packets will go to 2
	for(i = 0; i< MAX_IMAGES_BATCH_SIZE ;i++){
		image_to_instance_mapping[i] = 2;
	}

	const struct rte_memzone * services_memzone, *service_count_memzone;
	void *svr_addr;

	//attach to the services list from manager
	const char * services_info_mempool_name = "MProc_services_info";
	//open the mempool and extract 2 dimensional array.
	services_memzone = rte_memzone_lookup(services_info_mempool_name);

	svr_addr = services_memzone->addr;

	services = (uint16_t **) svr_addr;

	const char *services_count_address = "MProc_nf_per_service_info";
	service_count_memzone = rte_memzone_lookup(services_count_address);
	services_count = (uint16_t*)service_count_memzone->addr;
}



/* discover new services */
int discover_services(uint16_t** service_list, uint16_t* services_count);
int discover_services(uint16_t** service_list, uint16_t* services_count){
	int number_of_services_registered = services_count[preffered_service];
	uint8_t i = 0;
	printf("Number of Services: %d\n",number_of_services_registered);
	//now check which services are registered to our preferred service
	for(i=0; i<number_of_services_registered; i++){
		registered_services[i] = service_list[preffered_service][i];
		attach_to_shm(service_list[preffered_service][i]);
		open_mutex(service_list[preffered_service][i]);
	}
	//
	return 0;
}

//Special message handling function for loadbalancer
int check_load_balancing_messages(__attribute__((unused)) struct onvm_nf_msg *message_from_manager);
int check_load_balancing_messages(__attribute__((unused)) struct onvm_nf_msg *message_from_manager){

	//check and update the services
	discover_services(services, services_count);
	return 0;
}

/* main function */
int main(int argc, char *argv[]) {
        int arg_offset;

        const char *progname = argv[0];
        arg_offset = onvm_nflib_init(argc, argv, NF_TAG, &nf_info);

        if (arg_offset < 0)
                return -1;
        argc -= arg_offset;
        argv += arg_offset;

        if (parse_app_args(argc, argv, progname) < 0)
                rte_exit(EXIT_FAILURE, "Invalid command-line arguments\n");

        // Load the inference function mempools and put the address in the
        nfs = get_nfs(); //get all nfs variable
        register_gpu_msg_handling_function(check_load_balancing_messages); //registering a GPU msg handling function
        initialize_if_mempools();

        onvm_nflib_run(nf_info, &packet_handler);
        printf("If we reach here, program is ending");
        return 0;
}
