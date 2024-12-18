/*
 * python_c_api.c
 *
 *  Created on: Dec 23, 2019
 *      Author: root
 */
#include "/root/anaconda3/include/python3.7m/Python.h"
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <errno.h>
#include <limits.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>

#include "/home/adhak001/dev/openNetVM_sameer/onvm/onvm_nflib/onvm_ml_libraries.h"
#include "onvm_netml.h"

#include "python_c_api.h"

/* Py Objects for loading the file */
PyObject *pModule, *pDict,*myFileName, *mainFunc, *infer, *set_batch_size, *main_return, *set_batch_size_return, *infer_return;


/* code to initialize pytorch*/
int pytorch_link_model(__attribute__((unused)) nflib_ml_fw_link_params_t *load_params,__attribute__((unused)) void *aio)
{

       printf("Initializing PyTorch.. py is initialized %d \n", Py_IsInitialized());
	   char pythonhome[] = "PYTHONHOME=/root/anaconda3/";
	   putenv(pythonhome);

  
	   Py_SetPath(L"/home/adhak001/dev/openNetVM-sameer/examples/netml_ml_nf:/root/anaconda3/lib/python37.zip:/root/anaconda3/lib/python3.7:/root/anaconda3/lib/python3.7/lib-dynload:/root/anaconda3/lib/python3.7/site-packages:root/anaconda3/lib/python3.7/site-packages/torch:/root/anaconda3/lib/python3.7/site-packages/pretrainedmodels-0.7.4-py3.7.egg:/root/anaconda3/lib/python3.7/site-packages/munch-2.3.2-py3.7.egg:/root/anaconda3/lib/python3.7/site-packages/torchvision-0.2.1-py3.7.egg:/root/anaconda3/lib/python3.7/site-packages/onnx_cntk-1.0.0.0-py3.7.egg:/root/anaconda3/lib/python3.7/site-packages/apex-0.1-py3.7.egg:/home/adhak001/dev/clipper/clipper/clipper_admin");


	   
	   /* Initialize the python environment */
	   Py_Initialize();

	   /* sending a command line argument... */
	   wchar_t cmdline_w[] = L"python";
	   wchar_t *cmd = &cmdline_w[0];
	   PySys_SetArgvEx(1,&cmd,0);

	   
	printf("Initializing PyTorch.. py is initialized %d \n", Py_IsInitialized());
	const char * pyversion = Py_GetVersion();
	printf("Py version %s\n", pyversion);

	
	/* import libraries and set path before starting */
	 PyRun_SimpleString("import sys");
	 PyRun_SimpleString("sys.path.append(\".\")");

	 /* load the python file */
	 //myFileName = PyUnicode_FromString((const char *)"/home/adhak001/dev/DeepLearningExamples/PyTorch/Translation/GNMT/gnmt_python_c_interface");
	 PyRun_SimpleString("sys.path.append(\"/home/adhak001/dev/DeepLearningExamples/PyTorch/Translation/GNMT\")");
	 myFileName = PyUnicode_FromString((const char*)"gnmt_python_c_interface");
	 pModule = PyImport_Import(myFileName);

	 //If error Print
	 PyErr_Print();

	 /* load the name space of python program */
	 pDict = PyModule_GetDict(pModule);

	 /* Load all function names */
	 mainFunc = PyDict_GetItemString(pDict, (const char *)"main");
	 infer = PyDict_GetItemString(pDict, (const char *)"infer");
	 set_batch_size = PyDict_GetItemString(pDict, (const char *)"set_batch_size");

	 /* run the main funtion */
	 main_return = PyObject_CallObject(mainFunc,NULL);

	 /* test run.. feed the data
	 char hello[] = "hello dear\0";
	 char bye[] = "bye dear\0";
	 char hello_goodbye[] = "hello goodbye\0";

	 void * list_of_sentences[] = {hello, bye, hello_goodbye};
	 feed_data(3,list_of_sentences);
	 */
	 return 1;

}

/* a function to pack the sentences into python list */
//int feed_data(uint16_t num_of_sentences, void **array_of_sentences){

int pytorch_infer_batch(nflib_ml_fw_infer_params_t *load_params,__attribute__((unused))void *aio){

	//check how many batches of batches are
	int i = 0;
	printf("inferring batch size of %d\n",load_params->batch_size);
	for(i = 0; i<load_params->batch_size;i++){




//Remember don't forget to null terminate the string.. otherwise this can go bad very soon */
	uint32_t num_of_sentences = load_params->num_packets[i];
	void * array_of_sentences = (void *) load_params->array_of_packets[i];

  /* first we have to create string out of the packets  and append them to a list */
  PyObject *str, *sentence_list, *batch_size;
  uint32_t j = 0;

  batch_size = Py_BuildValue("(i)",num_of_sentences);
  //call the python function to set the batch size
  set_batch_size_return = PyObject_CallObject(set_batch_size, batch_size);
  //If error Print
  PyErr_Print();



  //Make a new list
  sentence_list = PyList_New(num_of_sentences);
  //iterate through the list of sentences
  for(j = 0; j<num_of_sentences; j++){
	  chunk_copy_info_t data_chunk = ((chunk_copy_info_t *) array_of_sentences)[j];
	      //void * src_ptr = data_chunk.src_cpy_ptr;
	      //uint32_t size = data_chunk.image_chunk.size_in_bytes;
	     // uint32_t offset = data_chunk.image_chunk.start_offset;
	      char *data = (char *) (data_chunk.src_cpy_ptr);
	      //printf("sentence %d : %s\n",j,data);
    size_t sentence_len = strlen(data);
    str = Py_BuildValue("s#",data,sentence_len);

    // Now push the string into a list
    PyList_Insert(sentence_list,j,str);
  }

  //printf("Calling to set batch_size\n");


  /* build the object for the list */
  PyObject *list_obj;
  list_obj = Py_BuildValue("(O)", sentence_list);
  infer_return = PyObject_CallObject(infer, list_obj);
  //If error Print
  PyErr_Print();

	}
  return 1;

}

int pytorch_load_model(__attribute__((unused)) nflib_ml_fw_load_params_t *load_params, __attribute__((unused)) void *aio){
	return 0;
}


/* lot of work needed to fix these modules.. 2 pytorch programs do not work the same so we rather have different functions for them */
int pytorch_load_model_yolo(__attribute__((unused)) nflib_ml_fw_load_params_t *load_params, __attribute__((unused)) void *aio){
	//No-op for now. pytorch_link_model_yolo takes care of model loading in GPU
	return 0;
}

/* the actual function to load the model into GPU */
int pytorch_link_model_yolo(nflib_ml_fw_link_params_t *load_params, __attribute__((unused)) void *aio){
	// start the GPU loading of the model
	printf("Initializing PyTorch.. py is initialized %d \n", Py_IsInitialized());
	char pythonhome[] = "PYTHONHOME=/root/anaconda3/";
	putenv(pythonhome);


	Py_SetPath(L"/home/adhak001/dev/openNetVM-sameer/examples/netml_ml_nf:/root/anaconda3/lib/python37.zip:/root/anaconda3/lib/python3.7:/root/anaconda3/lib/python3.7/lib-dynload:/root/anaconda3/lib/python3.7/site-packages:root/anaconda3/lib/python3.7/site-packages/torch:/root/anaconda3/lib/python3.7/site-packages/pretrainedmodels-0.7.4-py3.7.egg:/root/anaconda3/lib/python3.7/site-packages/munch-2.3.2-py3.7.egg:/root/anaconda3/lib/python3.7/site-packages/torchvision-0.2.1-py3.7.egg:/root/anaconda3/lib/python3.7/site-packages/onnx_cntk-1.0.0.0-py3.7.egg:/root/anaconda3/lib/python3.7/site-packages/apex-0.1-py3.7.egg:/home/adhak001/dev/clipper/clipper/clipper_admin");



	/* Initialize the python environment */
	Py_Initialize();

	/* sending a command line argument... */
	wchar_t cmdline_w[] = L"python";
	wchar_t *cmd = &cmdline_w[0];
	PySys_SetArgvEx(1,&cmd,0);


	printf("Initializing PyTorch.. py is initialized %d \n", Py_IsInitialized());
	const char * pyversion = Py_GetVersion();
	printf("Py version %s\n", pyversion);


	/* import libraries and set path before starting */
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append(\".\")");

	/* load the python file */
	//myFileName = PyUnicode_FromString((const char *)"/home/adhak001/dev/DeepLearningExamples/PyTorch/Translation/GNMT/gnmt_python_c_interface");
	PyRun_SimpleString("sys.path.append(\"/home/adhak001/dev/PyTorch-YOLOv3\")");
	myFileName = PyUnicode_FromString((const char*)"gslice_yolo_api");
	pModule = PyImport_Import(myFileName);

	//If error Print
	printf("Python Error:\n");
	PyErr_Print();

	/* load the name space of python program */
	pDict = PyModule_GetDict(pModule);


	/* Load all function names */
	mainFunc = PyDict_GetItemString(pDict, (const char *)"init_model");
	infer = PyDict_GetItemString(pDict, (const char *)"infer");
	//set_batch_size = PyDict_GetItemString(pDict, (const char *)"set_batch_size");

	/* run the main funtion */
	main_return = PyObject_CallObject(mainFunc,NULL);

	//If error Print
		printf("Python Error after init_model called:\n");
		PyErr_Print();

	//convert the return value into an address
	 long long gpu_buffer_address = PyLong_AsLongLong(main_return);

	load_params->gpu_side_input_pointer = (void*)gpu_buffer_address;

	//work done here, python is initialized and model is loaded
	return 0;
}


/* the function to infer the model */
int pytorch_infer_batch_yolo(nflib_ml_fw_infer_params_t* infer_params, __attribute__((unused)) void *aio){
	int batch_size = infer_params->batch_size;
	PyObject *batch_argument;
	batch_argument = Py_BuildValue("(i)",batch_size);
	PyObject_CallObject(infer,batch_argument);
	//If error Print
	//printf("Python Error after infer called:\n");
	PyErr_Print();

	return 0;
}
