/*
 * python_c_api.h
 *
 *  Created on: Dec 23, 2019
 *      Author: root
 */

#ifndef EXAMPLES_NETML_ML_NF_PYTHON_C_API_H_
#define EXAMPLES_NETML_ML_NF_PYTHON_C_API_H_
#include "/home/adhak001/dev/openNetVM_sameer/onvm/onvm_nflib/onvm_ml_libraries.h"
#ifdef __cplusplus
extern "C"{
#endif

  //int init_pytorch(void);
  //int feed_data(uint16_t num_of_sentences, void ** sentence_array);

int pytorch_load_model(nflib_ml_fw_load_params_t *load_params, void *aio);
/* the actual function to load the model into GPU */
int pytorch_link_model(nflib_ml_fw_link_params_t *load_params, void *aio);

/* the function to infer the model */
int pytorch_infer_batch(nflib_ml_fw_infer_params_t* infer_params, void *aio);

/* the function to get back the results if needed */
int pytorch_get_results(nflib_ml_fw_infer_params_t* infer_params, void *aio);

/* the deinitialization module */
int pytorch_deinit(uint32_t options);

#ifdef __cplusplus
}
#endif
  
#endif /* EXAMPLES_NETML_ML_NF_PYTHON_C_API_H_ */
