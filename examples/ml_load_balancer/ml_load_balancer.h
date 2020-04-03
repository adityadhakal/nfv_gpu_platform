/*
 * ml_load_balancer.h
 *
 *  Created on: Mar 17, 2020
 *      Author: Aditya Dhakal
 */
#ifndef EXAMPLES_ML_LOAD_BALANCER_ML_LOAD_BALANCER_H_
#define EXAMPLES_ML_LOAD_BALANCER_ML_LOAD_BALANCER_H_
#include "onvm_netml.h"

// 16 bit integer to hold info on which IF is currently subscribing to a service
uint16_t subscribed_if;


//The Shared memory addresses for IF's mutex to operate.
rte_atomic16_t *shm_server[MAX_NFS];

/* the mutex of all shm_servers */
sem_t *all_mutex[MAX_NFS];

/* the services list */
uint16_t ** services;
uint16_t * services_count;

uint16_t registered_services[MAX_NFS];

//a function that returns which IF this packet should go to: packet to IF mapping
int pkt_to_if(uint32_t packet_id);


//IF's load balancing data
typedef struct if_stats{
	uint8_t rate_of_request; // 1 sec / running average of latency
	struct timespec last_posting;//the time last image was posted
	histogram_v2_t latency_of_posting; // Running average of posting latency
	uint8_t weight; //how many percentage of request should go to this IF
}if_stats;

//for images that are already committeed to some IF
uint8_t committed_request[MAX_IMAGES_BATCH_SIZE];

#endif /* EXAMPLES_ML_LOAD_BALANCER_ML_LOAD_BALANCER_H_ */
