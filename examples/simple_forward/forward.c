/*********************************************************************
 *                     openNetVM
 *              https://sdnfv.github.io
 *
 *   BSD LICENSE
 *
 *   Copyright(c)
 *            2015-2016 George Washington University
 *            2015-2016 University of California Riverside
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided with the
 *       distribution.
 *     * Neither the name of Intel Corporation nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * monitor.c - an example using onvm. Print a message each p package received
 ********************************************************************/

#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdarg.h>
#include <errno.h>
#include <sys/queue.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>

#include <rte_common.h>
#include <rte_mbuf.h>
#include <rte_ip.h>
#include <rte_ether.h>

#include "onvm_nflib.h"
#include "onvm_pkt_helper.h"

#define NF_TAG "simple_forward"

/* Struct that contains information about this NF */
struct onvm_nf_info *nf_info;

/* number of package between each print */
static uint32_t print_delay = 5000000;


static uint32_t destination;

extern struct port_info *ports;
/*
 * Print a usage message
 */
static void
usage(const char *progname) {
        printf("Usage: %s [EAL args] -- [NF_LIB args] -- -d <destination> -p <print_delay>\n\n", progname);
}

/*
 * Parse the application arguments.
 */
static int
parse_app_args(int argc, char *argv[], const char *progname) {
        int c, dst_flag = 0;

        while ((c = getopt(argc, argv, "d:p:")) != -1) {
                switch (c) {
                case 'd':
                        destination = strtoul(optarg, NULL, 10);
                        dst_flag = 1;
                        break;
                case 'p':
                        print_delay = strtoul(optarg, NULL, 10);
                        break;
                case '?':
                        usage(progname);
                        if (optopt == 'd')
                                RTE_LOG(INFO, APP, "Option -%c requires an argument.\n", optopt);
                        else if (optopt == 'p')
                                RTE_LOG(INFO, APP, "Option -%c requires an argument.\n", optopt);
                        else if (isprint(optopt))
                                RTE_LOG(INFO, APP, "Unknown option `-%c'.\n", optopt);
                        else
                                RTE_LOG(INFO, APP, "Unknown option character `\\x%x'.\n", optopt);
                        return -1;
                default:
                        usage(progname);
                        return -1;
                }
        }

        if (!dst_flag) {
                RTE_LOG(INFO, APP, "Simple Forward NF requires destination flag -d.\n");
                return -1;
        }

        return optind;
}

/*
 * This function displays stats. It uses ANSI terminal codes to clear
 * screen when called. It is called from a single non-master
 * thread in the server process, when the process is run with more
 * than one lcore enabled.
 */
static void
do_stats_display(struct rte_mbuf* pkt) {
        const char clr[] = { 27, '[', '2', 'J', '\0' };
        const char topLeft[] = { 27, '[', '1', ';', '1', 'H', '\0' };
        static int pkt_process = 0;
        struct ipv4_hdr* ip;

        pkt_process += print_delay;

        /* Clear screen and move to top left */
        printf("%s%s", clr, topLeft);

        printf("PACKETS\n");
        printf("-----\n");
        printf("Port : %d\n", pkt->port);
        printf("Size : %d\n", pkt->pkt_len);
        printf("N°   : %d\n", pkt_process);
        printf("\n\n");

        ip = onvm_pkt_ipv4_hdr(pkt);
        if (ip != NULL) {
                onvm_pkt_print(pkt);
        } else {
                printf("No IP4 header found\n");
        }
}

void
do_check_and_insert_vlan_tag(struct rte_mbuf* pkt);
void
do_check_and_insert_vlan_tag(struct rte_mbuf* pkt) {
        /* This function will check if it is a valid ETH Packet
         * and if it is not a vlan_tagged, inserts a vlan tag
         */
        struct ether_hdr *eth = rte_pktmbuf_mtod(pkt, struct ether_hdr *);
        if (!eth) {
                exit(0);
                return ;
        }
        if (ETHER_TYPE_IPv4 == rte_be_to_cpu_16(eth->ether_type)) {
                if (rte_vlan_insert(&pkt)) {
                        printf("\nFailed to Insert Vlan Header to the Packet!!!!\n");
                        return;
                }
                struct vlan_hdr *vlan = (struct vlan_hdr*)(rte_pktmbuf_mtod(pkt, uint8_t*) + sizeof(struct ether_hdr));
                vlan->vlan_tci = rte_cpu_to_be_16((uint16_t)0x10);
                //vlan->eth_proto = rte_cpu_to_be_16(ETHER_TYPE_ARP);
                //printf("\nVLAN [0x%x, 0x%x] is already inserted!\n", rte_be_to_cpu_16(vlan->vlan_tci), rte_be_to_cpu_16(vlan->eth_proto));
        }
        else if (ETHER_TYPE_VLAN == rte_be_to_cpu_16(eth->ether_type)) {
                /*
                 struct vlan_hdr *vlan = (struct vlan_hdr*)(rte_pktmbuf_mtod(pkt, uint8_t*) + sizeof(struct ether_hdr));
                 if (vlan) {
                         printf("\nVLAN [0x%x, 0x%x] is already inserted!\n", rte_be_to_cpu_16(vlan->vlan_tci), rte_be_to_cpu_16(vlan->eth_proto));
                }
                */
        }
        else {
                printf("\nUnknown Ethernet Type [0x%x]!\n ", rte_be_to_cpu_16(eth->ether_type));
        }

        return;
}
#undef ENABLE_ND_MARKING_IN_NFS
#ifdef ENABLE_ND_MARKING_IN_NFS
/* Frequency of Non-determinism events : after every nondet_freq micro seconds */
//static uint32_t nondet_freq = (1000);
static uint64_t cycles_per_nd_mark = (3*1000*1000*100);  //10ms
//static uint64_t cycles_per_nd_mark =(nondet_freq*rte_get_timer_hz())/(1000*1000);
static volatile uint32_t nd_counter = 1;
static uint64_t last_cycle;
static uint64_t cur_cycle;
static int
callback_handler(void) {
        //return 0;
        cur_cycle = rte_get_tsc_cycles();
        uint64_t delta_cycles = cur_cycle - last_cycle;
        if (last_cycle && (((delta_cycles)) >=  cycles_per_nd_mark)) {
#ifdef ENABLE_LOCAL_LATENCY_PROFILER
                printf("Total elapsed cycles  %"PRIu64" (%"PRIu64" us) and packets before nd_sync: %" PRIu32 "\n", (delta_cycles),(((delta_cycles)*SECOND_TO_MICRO_SECOND)/rte_get_tsc_hz()), nd_counter);
#endif
                last_cycle = cur_cycle;
                nd_counter=0;
        }

        return 0;
}
#endif

static int
packet_handler(struct rte_mbuf* pkt, struct onvm_pkt_meta* meta) {
        static uint32_t counter = 0;
        if (++counter == print_delay) {
                do_stats_display(pkt);
                counter = 0;
        }
#ifdef ENABLE_ND_MARKING_IN_NFS
        if(nd_counter == 0) {
                meta->reserved_word |= NF_NEED_ND_SYNC;
                //printf("\n NF is raising ND Event!\n\n");
        } nd_counter++;
        if(0 == last_cycle) last_cycle = rte_get_tsc_cycles();
#endif

        //do_check_and_insert_vlan_tag(pkt);
        //if(0 == counter) do_stats_display(pkt);

        meta->action = ONVM_NF_ACTION_TONF;
        meta->destination = destination;


        if(likely(ports->num_ports > 1)) {
                meta->destination = (pkt->port == 0)? (PRIMARY_OUT_PORT):(0);
                if((PRIMARY_OUT_PORT == meta->destination) && (ports->down_status[PRIMARY_OUT_PORT])) {
                        meta->destination = SECONDARY_OUT_PORT;
                        //printf("Shifted traffic from primary out port sts=%d, to secondary out port\n", ports->down_status[PRIMARY_OUT_PORT]);
                }
        }
        meta->action = ONVM_NF_ACTION_OUT;
        //meta->destination = pkt->port;

        return 0;
}


int main(int argc, char *argv[]) {
        int arg_offset;

        const char *progname = argv[0];

        if ((arg_offset = onvm_nflib_init(argc, argv, NF_TAG)) < 0)
                return -1;
        argc -= arg_offset;
        argv += arg_offset;

        if (parse_app_args(argc, argv, progname) < 0)
                rte_exit(EXIT_FAILURE, "Invalid command-line arguments\n");

#ifndef ENABLE_ND_MARKING_IN_NFS
        onvm_nflib_run(nf_info, &packet_handler);
#else
        onvm_nflib_run_callback(nf_info, &packet_handler, &callback_handler);
#endif
        printf("If we reach here, program is ending");
        return 0;
}
