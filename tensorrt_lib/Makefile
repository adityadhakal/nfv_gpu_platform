CUDA_INSTALL_DIR=/usr/local/cuda
CURR_DIR=$(shell pwd)
CXXFLAGS=-std=c++11 -Wall -O3 -g -shared -fPIC -I$(CUDA_INSTALL_DIR)/include -I/usr/include/x86_64-linux-gnu/ -I$(CURR_DIR)/common -I/home/adhak001/dev/openNetVM_sameer/onvm/onvm_nflib/ -I/home/adhak001/tensorrt_binary/TensorRT/include
#CXXFLAGS=-std=c++11 -Wall -O3 -g -I$(CUDA_INSTALL_DIR)/include -I/usr/include/x86_64-linux-gnu/ -I$(CURR_DIR)/common -I/home/adhak001/dev/openNetVM_sameer/onvm/onvm_nflib/
LDFLAGS=-L$(CUDA_INSTALL_DIR)/lib64 -L$(CUDA_INSTALL_DIR)/lib64/stubs -L/usr/lib/x86_64-linux-gnu -L/home/adhak001/Downloads/TensorRT-7.0.0.11/lib
LDLIBS=-Wl,--start-group -lnvinfer -lcudart_static -lrt -ldl -lpthread -lstdc++ -lm -Wl,--end-group
CXX=g++

HEADERS=${wildcard *.h}
TARGET_SRCS=$(wildcard *.cpp)
TARGET=libtensorrt.so
#TARGET=tensorrt
OBJECTS=$(TARGET_SRCS: .cpp=.o)


all:$(TARGET)

$(TARGET):$(OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET) $(OBJECTS) $(LDLIBS)

clean:
	rm -r $(TARGETS) *.so
