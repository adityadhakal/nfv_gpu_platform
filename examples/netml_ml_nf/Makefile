#                    openNetVM
#      https://github.com/sdnfv/openNetVM
#
# BSD LICENSE
#
# Copyright(c)
#          2015-2017 George Washington University
#          2015-2017 University of California Riverside
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
APP = bridge

# all source are stored in SRCS-y
SRCS-y := bridge.c
SRCS-y += python_c_api.c

#SRCS-y:= /home/adhak001/dev/dlib/dlib/all/source.cpp
#SRCS-y += $(SRCDIR)/../../tensorrt_lib/tensorrt_test.o

ONVM= $(SRCDIR)/../../onvm


CFLAGS += -I/usr/local/cuda/include
LDFLAGS += -L/usr/local/cuda/lib64
LDFLAGS += -L/usr/local/cuda/lib64/stubs

#some python attachments
CFLAGS += -I/root/anaconda3/include/python3.7m
LDFLAGS += -L/root/anaconda3/lib/ -rpath,/root/anaconda3/lib/ -rpath,/root/anaconda3/lib/python37/



CFLAGS += -I/$(SRCDIR)/../../tensorrt_lib/

CFLAGS += -I/$(SRCDIR)/../../tensorrt_lib/common

LDFLAGS += $(SRCDIR)/../../tensorrt_lib/libtensorrt.so -rpath-link,/home/adhak001/dev/openNetVM_sameer/tensorrt_lib

CFLAGS += -I/usr/lib64
CFLAGS += -I/usr/include/x86_64-linux-gnu/
LDFLAGS += -L/usr/lib/x86_64-linux-gnu

CFLAGS += -I$(ONVM)/onvm_nflib
CFLAGS += -I$(ONVM)/lib

LDFLAGS += $(ONVM)/onvm_nflib/$(RTE_TARGET)/libonvm.a -lstdc++ -lcudart -lcuda
LDFLAGS += $(ONVM)/lib/$(RTE_TARGET)/lib/libonvmhelper.a
CFLAGS += -I/usr/local/include/
LDFLAGS += -ldlib -lX11 
#LDFLAGS += $(ONVM)/onvm_nflib/build/onvm_netml_cuda.o -lstdc++

#CFLAGS += -I$(SRCDIR)/../../cntk_lib/ 
#LDFLAGS += -L$(SRCDIR)/../../cntk_lib -L/home/adhak001/microsoft/cntk/build/release/lib -rpath-link,/home/adhak001/dev/openNetVM_sameer/cntk_lib -rpath-link,/home/adhak001/microsoft/cntk/build/release/lib/ -rpath,/home/adhak001/dev/openNetVM_sameer/cntk_lib -rpath,/home/adhak001/microsoft/cntk/build/release/lib
LDLIBS += -lnvinfer -lcudart_static -lrt -ldl -lpthread -lstdc++ -lm -lcudart

LDFLAGS += -lpython3.7m



LDFLAGS += $(RTE_SDK)/$(RTE_TARGET)/lib/librte_timer.a
CFLAGS += $(WERROR_FLAGS) -O3 -g -fPIC $(USER_FLAGS)



# workaround for a gcc bug with noreturn attribute
# http://gcc.gnu.org/bugzilla/show_bug.cgi?id=12603
ifeq ($(CONFIG_RTE_TOOLCHAIN_GCC),y)
CFLAGS_main.o += -Wno-return-type
endif

include $(RTE_SDK)/mk/rte.extapp.mk
