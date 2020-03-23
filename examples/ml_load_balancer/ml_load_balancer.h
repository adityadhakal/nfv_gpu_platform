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

// the array for image to instance mapping.. NOTE: This might have to be moved to the image data structure
uint8_t image_to_instance_mapping[MAX_IMAGES_BATCH_SIZE];

//The Shared memory addresses for IF's mutex to operate.
rte_atomic16_t *shm_server[MAX_NFS];

/* the mutex of all shm_servers */
sem_t *all_mutex[MAX_NFS];

/* the services list */
uint16_t ** services;
uint16_t * services_count;

uint16_t registered_services[MAX_NFS];

//a function that returns which IF this packet should go to: packet to IF mapping
uint8_t pkt_to_if(uint16_t packet_id);

//where should be packet be placed
uint8_t pkt_to_request(image_batched_aggregation_info_t *image_agg);

//for already committed images
uint8_t committed_request[MAX_IMAGES_BATCH_SIZE][MAX_NFS];

#endif /* EXAMPLES_ML_LOAD_BALANCER_ML_LOAD_BALANCER_H_ */
