/*
 * clipper_batchsize_extension.h
 *
 *  Created on: Dec 20, 2019
 *      Author: root
 */

#ifndef ONVM_ONVM_NFLIB_CLIPPER_BATCHSIZE_EXTENSION_H_
#define ONVM_ONVM_NFLIB_CLIPPER_BATCHSIZE_EXTENSION_H_

/* entry point to feed in batch statistics */
void clipper_add_processing_datapoint(size_t batch_size, long long processing_latency_micros);

/* one to get new batch size */
int clipper_check_batch_size(long deadline_us);

#endif /* ONVM_ONVM_NFLIB_CLIPPER_BATCHSIZE_EXTENSION_H_ */
