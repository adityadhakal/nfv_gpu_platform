#include "logger.h"
#include "common.h"
#include "argsParser.h"
#include "buffers.h"

#include "NvCaffeParser.h"
#include "NvInfer.h"

#include "cudaWrapper.h"
#include "ioHelper.h"
#include "tensorrt_api.h"

#include <inttypes.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

extern "C"{
#include "tensorrt_api.h"

}

using namespace cudawrapper;


// important file wide variables
ICudaEngine* engine{nullptr};

Logger gLogger;
// Declaring execution context.
unique_ptr<IExecutionContext, Destroy<IExecutionContext>> context{nullptr};
unique_ptr<IExecutionContext, Destroy<IExecutionContext>> context2{nullptr};
IExecutionContext *exe_context;

cudaStream_t stream; //single stream for now
void test_infer(IExecutionContext *context, void * bindings[2], cudaStream_t *cudastream);
//GPU side space to store.
void* bindings[2]{0};
void* bindings2[2]{0};
IRuntime *runtime_ptr;

/*--------------------------- end logging ----------------------- */
// Number of times we run inference to calculate average time.
/*
class Logger : public ILogger
{
  void log(Severity severity, const char* msg) override
  {
    // suppress info-level messages
    if (severity != Severity::kINFO)
      std::cout << msg << std::endl;
  }
} gLogger;
*/


// Returns empty string iff can't read the file
string readBuffer(string const& path)
{
  string buffer;
  ifstream stream(path.c_str(), ios::binary);

  if (stream)
    {
      stream >> noskipws;
      copy(istream_iterator<char>(stream), istream_iterator<char>(), back_inserter(buffer));
    }

  return buffer;
}
static int getBindingInputIndex(IExecutionContext* context)
{
  return !context->getEngine().bindingIsInput(0); // 0 (false) if bindingIsInput(0), 1 (true) otherwise
}

/* useless old junk code 
 *
 *
void launchInference(IExecutionContext* context, cudaStream_t stream, float* inputTensor, int inputTensorSize, float* outputTensor, int outputTensorSize,void** bindings, int batchSize)
{
  int inputId = getBindingInputIndex(context);

  cudaMemcpyAsync(bindings[inputId], inputTensor, inputTensorSize* sizeof(float), cudaMemcpyHostToDevice, stream);
  //cudaMemcpy(bindings[inputId], inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice);
  context->enqueue(batchSize, bindings, stream, nullptr);
  cudaMemcpyAsync(outputTensor, bindings[1 - inputId], outputTensorSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
  //cudaMemcpy(outputTensor.data(), bindings[1 - inputId], outputTensor.size() * sizeof(float), cudaMemcpyDeviceToHost);
}

//old function to launch inference
void launchInference_old(IExecutionContext* context, cudaStream_t stream, vector<float> const& inputTensor, vector<float>& outputTensor, void** bindings, int batchSize)
{
  int inputId = getBindingInputIndex(context);

  cudaMemcpyAsync(bindings[inputId], inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
  //cudaMemcpy(bindings[inputId], inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice);
  context->enqueue(batchSize, bindings, stream, nullptr);
  cudaMemcpyAsync(outputTensor.data(), bindings[1 - inputId], outputTensor.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
  //cudaMemcpy(outputTensor.data(), bindings[1 - inputId], outputTensor.size() * sizeof(float), cudaMemcpyDeviceToHost);
}


void doInference(IExecutionContext* context, cudaStream_t stream, vector<float> const& inputTensor, vector<float>& outputTensor, void** bindings, int batchSize)
{
  CudaEvent start;
  CudaEvent end;
  double totalTime = 0.0;
  struct timespec begin,finish;
  clock_gettime(CLOCK_MONOTONIC, &begin);
  int temp_batchsize = 1;
  for (int i = 0; i < ITERATIONS; ++i)
    {
      float elapsedTime;

      // Measure time it takes to copy input to GPU, run inference and move output back to CPU.
      cudaEventRecord(start, stream);
      launchInference_old(context, stream, inputTensor, outputTensor, bindings, batchSize);
      //launchInference(context, stream, inputTensor, outputTensor, bindings, temp_batchsize);
      cudaEventRecord(end, stream);

      // Wait until the work is finished.
      cudaStreamSynchronize(stream);
      cudaEventElapsedTime(&elapsedTime, start, end);

      totalTime += elapsedTime;
    }
  clock_gettime(CLOCK_MONOTONIC, &finish);

  double time_elapsed = (finish.tv_sec-begin.tv_sec)*1000.0+(finish.tv_nsec-begin.tv_nsec)/1000000.0;

  cout << "Inference batch size " << batchSize << " average over " << ITERATIONS << " runs is " << totalTime / ITERATIONS << "ms" << endl;
  std::cout<<" Inference of batch of "<< temp_batchsize << " number of iterations "<<ITERATIONS <<" total images "<< temp_batchsize*ITERATIONS<<" Time "<<time_elapsed<<" ms"<<" Throughput "<<(temp_batchsize*ITERATIONS)*1000/time_elapsed<<endl;
}
*/
/* Tensorrt does nothing with load function */
extern "C"
int tensorrt_load_model(nflib_ml_fw_load_params_t *load_params, void *aio){
	//string buffer = readBuffer("/usr/src/tensorrt/bin/resnet50_batch64.trt");
	if(load_params->load_options==1){
	  string buffer = readBuffer(load_params->file_path);
	  std::cout<<"Buffer size "<<buffer.size()<<std::endl;
	  memcpy(load_params->ml_file_buffer,(void *)buffer.data(),buffer.size());
	  return 0;
	}
}


/* tensorrt performs loading with link function */
extern "C"
int tensorrt_link_model(nflib_ml_fw_link_params_t *link_params, void *ml)
//int main()
{
  struct timespec begin,end, load_end,deserialize_end;
  clock_gettime(CLOCK_MONOTONIC, &begin);
  
  void *string_buffer;
  size_t buffer_size;
  string buffer;
  //string buffer = readBuffer("/usr/src/tensorrt/bin/resnet50_batch64.trt");
  if(link_params->model_buffer_size>0){
	string_buffer = link_params->ml_file_buffer;
	buffer_size = (size_t) link_params->model_buffer_size;
  }
  else{
	  buffer = readBuffer(link_params->file_path);
	  string_buffer = (void *)buffer.data();
	  buffer_size = buffer.size();
  }
  std::cout<<"Buffer size "<<buffer_size<<std::endl;

  clock_gettime(CLOCK_MONOTONIC, &load_end);
  double load_time = (load_end.tv_sec-begin.tv_sec)*1000.0+(load_end.tv_nsec-begin.tv_nsec)/1000000.0;
  std::cout<<"File Load Time: (ms) "<<load_time<<std::endl;

  if (buffer_size)
    {
      // try to deserialize engine
      unique_ptr<IRuntime, Destroy<IRuntime>> runtime{createInferRuntime(gLogger)};
      runtime_ptr = runtime.get();
      engine = runtime->deserializeCudaEngine(string_buffer, buffer_size, nullptr);
    }

  
  // Assume networks takes exactly 1 input tensor and outputs 1 tensor.
  assert(engine->getNbBindings() == 2);
  assert(engine->bindingIsInput(0) ^ engine->bindingIsInput(1));

  clock_gettime(CLOCK_MONOTONIC, &deserialize_end);
  double deserialize_time = (deserialize_end.tv_sec-load_end.tv_sec)*1000.0+(deserialize_end.tv_nsec-load_end.tv_nsec)/1000000.0;
  std::cout<<"Deserialize time (ms) : "<<deserialize_time<<std::endl;
  
  /*
  vector<float> inputTensor;
  vector<float> outputTensor;
  int batchSize = 64;
  for (int i = 0; i < engine->getNbBindings(); ++i)
    {
      Dims dims{engine->getBindingDimensions(i)};
      size_t size = accumulate(dims.d, dims.d + dims.nbDims, batchSize, multiplies<size_t>());

      printf("Size of dimension %d %ld\n",i, size);
      // Create CUDA buffer for Tensor.
      cudaMalloc(&bindings[i], size * sizeof(float));

      // Resize CPU buffers to fit Tensor.
     
      if (engine->bindingIsInput(i))
	inputTensor.resize(size);
      else
	outputTensor.resize(size);
      
    }
  
    srand(0);
  
  for(int i = 0; i<inputTensor.size(); i++){
    inputTensor[i] = ((rand()%255)/255.0);
  }

  cudaHostRegister(inputTensor.data(), inputTensor.size()*sizeof(float), cudaHostRegisterDefault);
  cudaHostRegister(outputTensor.data(), outputTensor.size()*sizeof(float), cudaHostRegisterDefault);
    
    */
  // Create Execution Context.
  context.reset(engine->createExecutionContext());
  context2.reset(engine->createExecutionContext());
  /* return the context created so it can be stored for future inferences */
  exe_context = context.get();

/*
  int inputId = getBindingInputIndex(exe_context);
  //cudaMemcpy(bindings[inputId],inputTensor.data(), (inputTensor.size() * sizeof(float)), cudaMemcpyHostToDevice);

  cudaStream_t stream[4];
  for(int i = 0; i <4; i++){
    cudaStreamCreateWithFlags(&stream[i],cudaStreamNonBlocking);
  }
  //test_infer(exe_context);
  printf("Testing inference \n");
  for(int i = 0 ; i<100;i++){
    test_infer(exe_context, bindings, &stream[i%4]);
  }
*/
  //int inputId = getBindingInputIndex(context);
  //cudaMemcpyAsync(bindings[inputId], inputTensor, inputTensorSize* sizeof(float), cudaMemcpyHostToDevice, stream);
  //cudaMemcpy(bindings[inputId], inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice);

  //exe_context->enqueue(infer_params->batch_size, bindings, *infer_params->stream, nullptr);
  
  //exe_context->enqueue(1, bindings, 0, nullptr);
  //exe_context->execute(1,bindings);
  //cudaMemcpy(outputTensor.data(),bindings[1-inputId], (outputTensor.size() * sizeof(float)), cudaMemcpyDeviceToHost);
  //cudaDeviceSynchronize();
  /*
  std::cout<<"Inference finished \n";
  for(int i = 0; i<100; i++){
    std::cout<<" "<<inputTensor[i];
   
    std::cout<<" "<<outputTensor[i]<<std::endl;
  }
  */
  clock_gettime(CLOCK_MONOTONIC, &end);
  float time_spent = (end.tv_sec-begin.tv_sec)*1000+(end.tv_nsec-begin.tv_nsec)/1000000;
  printf("Time taken to load the model is %f milliseconds\n", time_spent);
  uint64_t microsec_timestp;
  microsec_timestp = end.tv_sec*1000000+end.tv_nsec/1000;
  printf("Model load timestamp %"PRIu64"\n",microsec_timestp);
  return 0;
}

/*
cudaStreamCallback_t callback_function(cudaStream_t stream, void* a);
cudaStreamCallback_t callback_function(void* a){
  printf("Callback called...\n");
  
  
}
*/
void callback_function(cudaStream_t stream, cudaError_t status, void* a);
void callback_function(cudaStream_t stream, cudaError_t status, void* a){
  printf("Callback called...\n");
  
  
}


void test_infer(IExecutionContext *context, void *bindings[2],cudaStream_t *stream){
  printf("Testing inference \n");
  /*int batchSize = 64;
  for (int i = 0; i < engine->getNbBindings(); ++i)
    {
      Dims dims{engine->getBindingDimensions(i)};
      size_t size = accumulate(dims.d, dims.d + dims.nbDims, batchSize, multiplies<size_t>());
      // Create CUDA buffer for Tensor.
      cudaMalloc(&bindings[i], size * sizeof(float));
    }
  */
  //bindings[0] = input;
  //bindings[1] = output;


  //int inputId = getBindingInputIndex(context);
  //cudaMemcpyAsync(bindings[inputId], inputTensor, inputTensorSize* sizeof(float), cudaMemcpyHostToDevice, stream);
  //cudaMemcpy(bindings[inputId], inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice);

  context2->enqueue(1, bindings, *stream, nullptr);
  //context->enqueue(1, bindings, 0, nullptr);
  /*  
  int aa = 1;
  void * a = &aa;
  cudaStreamAddCallback(*stream, callback_function, a,0);
  */
}
/* evaluate a batch of images */
int tensorrt_infer_batch(nflib_ml_fw_infer_params_t *infer_params, void *aio){

  //if(aio != NULL){
  

  //we have to check the incoming batch size
  //std::cout<<"Batch size: "<<infer_params->batch_size<<std::endl;

  //printf("Input buffer %p and output %p \n",infer_params->input_data, infer_params->output);
  //IExecutionContext * context = (IExecutionContext*) infer_params->model_handle;
 
  //void *input,*output;
  //cudaMalloc(&input, sizeof(float)*3*224*224);
  //cudaMalloc(&output, sizeof(float)*1000);

  bindings[0] = infer_params->input_data;
  bindings[1] = infer_params->output;

  //  printf("size of input %ld and size of batch %d n",infer_params->input_size, infer_params->batch_size);
  
 /*
  int batchSize = 64;
  for (int i = 0; i < engine->getNbBindings(); ++i)
    {
      Dims dims{engine->getBindingDimensions(i)};
      size_t size = accumulate(dims.d, dims.d + dims.nbDims, batchSize, multiplies<size_t>());
      printf("new engine Size %ld \n",size);
      // Create CUDA buffer for Tensor.
      cudaMalloc(&bindings2[i], size * sizeof(float));
    }


  vector<float> inputTensor;
  inputTensor.resize((3*224*224));
      //bindings[0] = input;
      //bindings[1] = output;
    // Create Execution Context.
  //context.reset(engine->createExecutionContext());

  srand(0);
  printf("size of input tensor %ld \n", inputTensor.size());
  for(int i = 0; i<inputTensor.size(); i++){
    inputTensor[i] = ((rand()%255)/255.0);
  }

  cudaMemcpy(bindings2[0],inputTensor.data(), inputTensor.size()*sizeof(float),cudaMemcpyHostToDevice);
  float a[100];
  float b[100];
  cudaMemcpy(a, bindings2[0], sizeof(float)*100, cudaMemcpyDeviceToHost);
     for(int i = 0; i< 100; i++)
     {
       std::cout<<a[i]<<" "<<b[i]<<std::endl;
     }
*/
  /* return the context created so it can be stored for future inferences */
  //exe_context = context.get();
    // Create Execution Context.
  //context2.reset(engine->createExecutionContext());
  
  
  //int inputId = getBindingInputIndex(context);
  //cudaMemcpyAsync(bindings[inputId], inputTensor, inputTensorSize* sizeof(float), cudaMemcpyHostToDevice, stream);
  //cudaMemcpy(bindings[inputId], inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice);
   
  //static int a = 0;
  /* the execution part */
  //if(a %2 == 0){
    context->enqueue(infer_params->batch_size, bindings, *(infer_params->stream), nullptr);
  //}
  //if(a %2 == 1){
  //  context2->enqueue(infer_params->batch_size, bindings, *(infer_params->stream), nullptr);
  //}

//  a++;
  //context2.get()->enqueue(infer_params->batch_size, bindings2, 0, nullptr);
  //context.get()->execute(1,bindings);
  //cudaMemcpyAsync(outputTensor, bindings[1 - inputId], outputTensorSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
  //cudaMemcpy(outputTensor.data(), bindings[1 - inputId], outputTensor.size() * sizeof(float), cudaMemcpyDeviceToHost);

  //cudaDeviceSynchronize();
  //let's check the input
  //cudaMemcpy(b, bindings2[1], sizeof(float)*100, cudaMemcpyDeviceToHost);


/*
  if(infer_params->callback_function !=NULL){
    cudaLaunchHostFunc(*(infer_params->stream), infer_params->callback_function, infer_params->callback_data);
    //cudaLaunchHostFunc(0, infer_params->callback_function, infer_params->callback_data);
  }
*/
  

  return 0;
}

