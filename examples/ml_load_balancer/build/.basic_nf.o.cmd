cmd_basic_nf.o = gcc -Wp,-MD,./.basic_nf.o.d.tmp  -m64 -pthread  -march=native -mno-avx512f -DRTE_MACHINE_CPUFLAG_SSE -DRTE_MACHINE_CPUFLAG_SSE2 -DRTE_MACHINE_CPUFLAG_SSE3 -DRTE_MACHINE_CPUFLAG_SSSE3 -DRTE_MACHINE_CPUFLAG_SSE4_1 -DRTE_MACHINE_CPUFLAG_SSE4_2 -DRTE_MACHINE_CPUFLAG_AES -DRTE_MACHINE_CPUFLAG_PCLMULQDQ -DRTE_MACHINE_CPUFLAG_AVX -DRTE_MACHINE_CPUFLAG_RDRAND -DRTE_MACHINE_CPUFLAG_FSGSBASE -DRTE_MACHINE_CPUFLAG_F16C -DRTE_MACHINE_CPUFLAG_AVX2  -I/home/adhak001/dev/openNetVM_sameer/examples/ml_load_balancer/build/include -I/home/skulk901/dev/openNetVM_Mainline/dpdk/x86_64-native-linuxapp-gcc/include -include /home/skulk901/dev/openNetVM_Mainline/dpdk/x86_64-native-linuxapp-gcc/include/rte_config.h -D_GNU_SOURCE -W -Wall -Wstrict-prototypes -Wmissing-prototypes -Wmissing-declarations -Wold-style-definition -Wpointer-arith -Wcast-align -Wnested-externs -Wcast-qual -Wformat-nonliteral -Wformat-security -Wundef -Wwrite-strings -Wdeprecated -Werror -O3 -g -I/home/adhak001/dev/openNetVM_sameer/examples/ml_load_balancer/../../onvm/onvm_nflib -I/home/adhak001/dev/openNetVM_sameer/examples/ml_load_balancer/../../onvm/shared -I/home/adhak001/dev/openNetVM_sameer/examples/ml_load_balancer/../../onvm/lib -I/usr/local/cuda/include    -o basic_nf.o -c /home/adhak001/dev/openNetVM_sameer/examples/ml_load_balancer/basic_nf.c 
