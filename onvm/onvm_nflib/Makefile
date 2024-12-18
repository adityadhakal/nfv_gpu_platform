#                    openNetVM
#      https://github.com/sdnfv/openNetVM
#
# BSD LICENSE
#
# Copyright(c)
#          2015-2017 George Washington University
#          2015-2017 University of California Riverside
#          2010-2014 Intel Corporation.
#          2016-2017 Hewlett Packard Enterprise Development LP
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in
# the documentation and/or other materials provided with the
# distribution.
# The name of the author may not be used to endorse or promote
# products derived from this software without specific prior
# written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

ifeq ($(RTE_SDK),)
$(error "Please define RTE_SDK environment variable")
endif

# Default target, can be overriden by command line or environment
include $(RTE_SDK)/mk/rte.vars.mk

RTE_TARGET ?= x86_64-native-linuxapp-gcc

# binary name
LIB    = libonvm.a

# all source are stored in SRCS-y
SRCS-y := onvm_pkt_helper.c onvm_sc_common.c onvm_sc_mgr.c onvm_flow_table.c onvm_flow_dir.c onvm_nflib.c
#SRCS-y += onvm_pkt_common.c
SRCS-y += onvm_netml_cuda.o
SRCS-y += clipper_batchsize_extension.o
SRCS-y += onvm_gpu_buffer_factory.c onvm_stream.c
SRCS-y += onvm_netml.c
SRCS-y += onvm_config_common.c
SRCS-y += histogram.c
SRCS-y += onvm_sort.c
SRCS-y += onvm_ringbuf.c
SRCS-y += onvm_comm_utils.c
CFLAGS += -DUSE_HISTOGRAM_AS_LIB


CFLAGS += $(WERROR_FLAGS) -O3 -lstdc++ -g $(USER_FLAGS)
CFLAGS += -I$(SRCDIR)/../lib/
#CFLAGS += -I$(ONVM_HOME)/onvm/lib/
CFLAGS += -I$(RTE_SDK)/$(RTE_TARGET)/include/
LDFLAGS += $(RTE_SDK)/$(RTE_TARGET)/lib/librte_timer.a
CFLAGS += -I/usr/local/cuda/include
LDFLAGS += -L/usr/local/cuda/lib64

LDFLAGS += -L/usr/local/lib

CFLAGS += -I$(SRCDIR)/../../tensorrt_lib/
LDFLAGS += $(SRCDIR)/../../tensorrt_lib/libtensorrt.so 
LDLIBS += -lzmq -lstdc++ -lm 
LDLIBS += -lonvm_cntk -lcudart


CFLAGS += -I$(SRCDIR)/../../cntk_lib/ #-Wl, #-Wl,-rpath-link,/home/adhak001/microsoft/cntk/build/release/lib/
LDFLAGS += -L$(SRCDIR)/../../cntk_lib -L/home/adhak001/microsoft/cntk/build/release/lib -rpath-link,/home/adhak001/dev/openNetVM_sameer/cntk_lib -rpath-link,/home/adhak001/microsoft/cntk/build/release/lib/ -rpath,/home/adhak001/dev/openNetVM_sameer/cntk_lib -rpath,/home/adhak001/microsoft/cntk/build/release/lib


LDFLAGS += ./libzmq.a


include $(RTE_SDK)/mk/rte.extlib.mk

onvm_netml_cuda.o: onvm_netml_cuda.cu
	nvcc -c -g $(SRCDIR)/onvm_netml_cuda.cu -lstdc++

clipper_batchsize_extension.o: clipper_batchsize_extension.cpp
	gcc -lstdc++ -c $(SRCDIR)/clipper_batchsize_extension.cpp /home/adhak001/dev/dlib/dlib/all/source.cpp -std=c++11 -O3 -ldlib -lX11 -lpthread -I /home/adhak001/dev/dlib/
