/*********************************************************************
 *                     openNetVM
 *              https://sdnfv.github.io
 *
 *   BSD LICENSE
 *
 *   Copyright(c)
 *            2015-2017 George Washington University
 *            2015-2017 University of California Riverside
 *            2016-2017 Hewlett Packard Enterprise Development LP
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
 *     * The name of the author may not be used to endorse or promote
 *       products derived from this software without specific prior
 *       written permission.
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
 ********************************************************************/

/******************************************************************************

                                  onvm_nflib.c


                  File containing all functions of the NF API


******************************************************************************/


/***************************Standard C library********************************/


#include <getopt.h>
#include <signal.h>

/******************************DPDK libraries*********************************/
#include "rte_malloc.h"

/*****************************Internal headers********************************/


#include "onvm_nflib_internal.h"
#include "onvm_nflib.h"
#ifdef ONVM_GPU
#include <cuda_runtime.h>
#include "onvm_images.h"
#endif
#include "onvm_includes.h"
#include "onvm_sc_common.h"

/**********************************Macros*************************************/


// Possible NF packet consuming modes
#define NF_MODE_UNKNOWN 0
#define NF_MODE_SINGLE 1
#define NF_MODE_RING 2

#define ONVM_NO_CALLBACK NULL

#define ENABLE_NF_PAUSE_TILL_OUTSTANDING_NDSYNC_COMMIT
#define __DEBUG__NDSYNC_LOGS__

#ifdef ENABLE_NF_PAUSE_TILL_OUTSTANDING_NDSYNC_COMMIT
#ifdef  __DEBUG_NDSYNC_LOGS__
onvm_interval_timer_t nd_ts;
int64_t max_nd, min_nd, avg_nd, delta_nd;
#endif
#endif
/*********************************************************************/
/*            NF LIB Feature flags specific functions                */
/*********************************************************************/

#ifdef TEST_MEMCPY_OVERHEAD
static inline void allocate_base_memory(void) {
        base_memory = calloc(1,2*MEMCPY_SIZE);
}
static inline void do_memcopy(void *from_pointer) {
        if(likely(base_memory && from_pointer)) {
                memcpy(base_memory,from_pointer,MEMCPY_SIZE);
        }
}
#endif //TEST_MEMCPY_OVERHEAD

/*********************************************************************/
nf_explicit_callback_function nf_ecb = NULL;
static uint8_t need_ecb = 0;
void register_explicit_callback_function(nf_explicit_callback_function ecb) {
        if(ecb) {
                nf_ecb = ecb;
        }
        return;
}
/******************************************************************************/
/*                        HISTOGRAM DETAILS                                   */
/******************************************************************************/

/*********************** ADITYA'S CODE ****************************************/
/**************** ONVM_GPU ******************/
#ifdef ONVM_GPU
gpu_message_processing_func nf_gpu_func;
void register_gpu_msg_handling_function(gpu_message_processing_func gmpf){
  if(gmpf){
    nf_gpu_func = gmpf;
  }
  return;
}
//timer threads...
void initialize_ml_timers(struct onvm_nf_info *nf_info);


#endif

/************************************API**************************************/

#ifdef USE_CGROUPS_PER_NF_INSTANCE
#include <stdlib.h>
uint32_t get_nf_core_id(void);
void init_cgroup_info(struct onvm_nf_info *nf_info);
int set_cgroup_cpu_share(struct onvm_nf_info *nf_info, unsigned int share_val);

uint32_t get_nf_core_id(void) {
        return rte_lcore_id();
}

int set_cgroup_cpu_share(struct onvm_nf_info *nf_info, unsigned int share_val) {
        /*
        unsigned long shared_bw_val = (share_val== 0) ?(1024):(1024*share_val/100); //when share_val is relative(%)
        if (share_val >=100) {
                shared_bw_val = shared_bw_val/100;
        }*/

        unsigned long shared_bw_val = (share_val== 0) ?(1024):(share_val);  //when share_val is absolute bandwidth
        const char* cg_set_cmd = get_cgroup_set_cpu_share_cmd(nf_info->instance_id, shared_bw_val);
        printf("\n CMD_TO_SET_CPU_SHARE: %s", cg_set_cmd);
        int ret = system(cg_set_cmd);
        if  (0 == ret) {
                nf_info->cpu_share = shared_bw_val;
        }
        return ret;
}
void init_cgroup_info(struct onvm_nf_info *nf_info) {
        int ret = 0;
        const char* cg_name = get_cgroup_name(nf_info->instance_id);
        const char* cg_path = get_cgroup_path(nf_info->instance_id);
        printf("\n NF cgroup name and path: %s, %s", cg_name,cg_path);

        /* Check and create the CGROUP if necessary */
        const char* cg_crt_cmd = get_cgroup_create_cgroup_cmd(nf_info->instance_id);
        printf("\n CMD_TO_CREATE_CGROUP_for_NF: %d, %s", nf_info->instance_id, cg_crt_cmd);
        ret = system(cg_crt_cmd);

        /* Add the pid to the CGROUP */
        const char* cg_add_cmd = get_cgroup_add_task_cmd(nf_info->instance_id, nf_info->pid);
        printf("\n CMD_TO_ADD_NF_TO_CGROUP: %s", cg_add_cmd);
        ret = system(cg_add_cmd);

        /* Retrieve the mapped core-id */
        nf_info->core_id = get_nf_core_id();

        /* Initialize the cpu.shares to default value (100%) */
        ret = set_cgroup_cpu_share(nf_info, 0);

        printf("NF on core=%u added to cgroup: %s, ret=%d", nf_info->core_id, cg_name,ret);
        return;
}
#endif //USE_CGROUPS_PER_NF_INSTANCE

/******************************FTMB Helper Functions********************************/
#ifdef MIMIC_FTMB
//#define SV_ACCES_PER_PACKET (2)   //moved as configurable variable.
typedef struct pal_packet {
        uint64_t pal_counter;
        uint64_t pal_info;
}pal_packet_t;
static inline int send_packets_out(uint8_t port_id, uint16_t queue_id, struct rte_mbuf **tx_pkts, uint16_t nb_pkts) {

        return 0;
        uint16_t sent_packets = 0; //rte_eth_tx_burst(port_id,queue_id, tx_pkts, nb_pkts);
        if(unlikely(sent_packets < nb_pkts)) {
                uint16_t i = sent_packets;
                for(; i< nb_pkts;i++)
                        onvm_nflib_drop_pkt(tx_pkts[i]);
        }
/*
        {
                volatile struct tx_stats *tx_stats = &(ports->tx_stats);
                tx_stats->tx[port_id] += sent_packets;
                tx_stats->tx_drop[port_id] += (nb_pkts - sent_packets);
        }
*/
        return sent_packets;
        rte_eth_tx_burst(port_id,queue_id, tx_pkts, nb_pkts);
}
int generate_and_transmit_pals_for_batch(__attribute__((unused))  void **pktsTX, unsigned tx_batch_size, __attribute__((unused)) unsigned non_det_evt, __attribute__((unused)) uint64_t ts_info);
int generate_and_transmit_pals_for_batch(__attribute__((unused))  void **pktsTX, unsigned tx_batch_size, __attribute__((unused)) unsigned non_det_evt, __attribute__((unused)) uint64_t ts_info) {
        uint32_t i = 0;
        static uint64_t pal_counter = 0;
        struct rte_mbuf *out_pkt = NULL;
        struct onvm_pkt_meta *pmeta = NULL;
        struct ether_hdr *eth_hdr = NULL;
        pal_packet_t *p_hdr;
        size_t pkt_size = 0;
        size_t data_len = sizeof(struct ether_hdr) + sizeof(uint16_t) + sizeof(uint16_t);
        int ret = 0;
        //void *pkts[CLIENT_SHADOW_RING_SIZE];
#ifdef ENABLE_LOCAL_LATENCY_PROFILER
        static uint64_t pktcounterm=0; uint64_t start_cycle=0;onvm_interval_timer_t ts_p;
        pktcounterm+=(tx_batch_size*SV_ACCES_PER_PACKET);
        if(pktcounterm >= (1000*1000*20)) {
                onvm_util_get_start_time(&ts_p);
                start_cycle = onvm_util_get_current_cpu_cycles();
        }
#endif
        //for(i=0; i<(SV_ACCES_PER_PACKET); i++) {
        for(i=0; i<(tx_batch_size*SV_ACCES_PER_PACKET); i++) {

                //Allocate New Packet
                out_pkt = rte_pktmbuf_alloc(pktmbuf_pool);
                if (out_pkt == NULL) {
                        rte_pktmbuf_free(out_pkt);
                        //rte_exit(EXIT_FAILURE, "onvm_nflib:Failed to alloc packet for %x, %li!! \n", i, pal_counter);
                        //RTE_LOG(ERROR, APP, "onvm_nflib:Failed to alloc packet for %x, %li!! \n", i, pal_counter);
                        fprintf(stderr, "onvm_nflib:Failed to alloc packet for %x, %li!! \n", i, pal_counter);
                        return -1;
                }

                //set packet properties
                pkt_size = sizeof(struct ether_hdr) + sizeof(pal_packet_t);
                out_pkt->data_len = MAX(pkt_size, data_len);    //todo: crirical error if 0 or lesser than pkt_len: mooongen discards; check again and confirm
                out_pkt->pkt_len = pkt_size;

                //Set Ethernet Header info
                eth_hdr = onvm_pkt_ether_hdr(out_pkt);
                eth_hdr->ether_type = rte_cpu_to_be_16((ETHER_TYPE_RSYNC_DATA+1));
                //ether_addr_copy(&ports->mac[port], &eth_hdr->s_addr);
                //ether_addr_copy((const struct ether_addr *)&rsync_node_info.mac_addr_bytes, &eth_hdr->d_addr);

                //SET PAL DATA
                p_hdr = rte_pktmbuf_mtod_offset(out_pkt, pal_packet_t*, sizeof(struct ether_hdr));
                p_hdr->pal_counter = pal_counter++;
                p_hdr->pal_info = 0xBADABADBBADCBADD;


                //SEND PACKET OUT/SET METAINFO
                pmeta = onvm_get_pkt_meta(out_pkt);
                pmeta->destination =RSYNC_TX_OUT_PORT;
                pmeta->action = ONVM_NF_ACTION_DROP;

                send_packets_out(0, RSYNC_TX_PORT_QUEUE_ID_0, &out_pkt, 1);
                if(unlikely(-ENOBUFS == (ret = rte_ring_enqueue(tx_ring, out_pkt)))) {
                        do {
#ifdef INTERRUPT_SEM
                                onvm_nf_yeild(nf_info,YIELD_DUE_TO_FULL_TX_RING);
                                ret = rte_ring_enqueue(tx_ring, out_pkt);
#endif
                        }while((ret == -ENOBUFS) && keep_running);
                }
        }
#ifdef ENABLE_LOCAL_LATENCY_PROFILER
        if((pktcounterm)>=(1000*1000*20)) {
                fprintf(stdout, "PAL GENERATION TIME for (%x) SV : %li(ns) and %li (cycles) for packet:%d \n", SV_ACCES_PER_PACKET, onvm_util_get_elapsed_time(&ts_p), onvm_util_get_elapsed_cpu_cycles(start_cycle), (int)pkt_size);
                pktcounterm=0;
        }
#endif
        return ret;
}
#endif
/******************************Timer Helper functions*******************************/
#ifdef ENABLE_TIMER_BASED_NF_CYCLE_COMPUTATION
static void
stats_timer_cb(__attribute__((unused)) struct rte_timer *ptr_timer,
        __attribute__((unused)) void *ptr_data) {

#ifdef INTERRUPT_SEM
        counter = SAMPLING_RATE;
#endif //INTERRUPT_SEM

        //printf("\n On core [%d] Inside Timer Callback function: %"PRIu64" !!\n", rte_lcore_id(), rte_rdtsc_precise());
        //printf("Echo %d", system("echo > hello_timer.txt"));
        //printf("\n Inside Timer Callback function: %"PRIu64" !!\n", rte_rdtsc_precise());
}

static inline void
init_nflib_timers(void) {
        //unsigned cur_lcore = rte_lcore_id();
        //unsigned timer_core = rte_get_next_lcore(cur_lcore, 1, 1);
        //printf("cur_core [%u], timer_core [%u]", cur_lcore,timer_core);
        rte_timer_subsystem_init();
        rte_timer_init(&stats_timer);
        rte_timer_reset_sync(&stats_timer,
                                (NF_STATS_PERIOD_IN_MS * rte_get_timer_hz()) / 1000,
                                PERIODICAL,
                                rte_lcore_id(), //timer_core
                                &stats_timer_cb, NULL
                                );
}
#endif
/******************************Timer Helper functions*******************************/
#ifdef INTERRUPT_SEM
void onvm_nf_yeild(__attribute__((unused))struct onvm_nf_info* info, __attribute__((unused)) uint8_t reason_rxtx) {
        
        /* For now discard the special NF instance and put all NFs to wait */
       // if ((!ONVM_SPECIAL_NF) || (info->instance_id != 1)) { }

#ifdef ENABLE_NF_YIELD_NOTIFICATION_COUNTER
        if(reason_rxtx) {
                this_nf->stats.tx_drop+=1;
        }else {
                this_nf->stats.yield_count +=1;
        }
#endif

#ifdef USE_POLL_MODE
        return;
#endif

        //do not block if running status is off.
        if(unlikely(!keep_running)) return;

        rte_atomic16_set(flag_p, 1);  //rte_atomic16_cmpset(flag_p, 0, 1);
#ifdef USE_SEMAPHORE
        sem_wait(mutex);
#endif
        //rte_atomic16_set(flag_p, 0); //out of block rte_atomic16_cmpset(flag_p, 1, 0);
        
        //check and trigger explicit callabck before returning.
        if(need_ecb && nf_ecb) {
                need_ecb = 0;
                nf_ecb();
        }
}
#ifdef INTERRUPT_SEM
static inline void  onvm_nf_wake_notify(__attribute__((unused))struct onvm_nf_info* info);
static inline void  onvm_nf_wake_notify(__attribute__((unused))struct onvm_nf_info* info)
{
#ifdef USE_SEMAPHORE
        sem_post(mutex);
        //printf("Triggered to wakeup the NF thread internally");
#endif
        return;
}
static inline void onvm_nflib_implicit_wakeup(void);
static inline void onvm_nflib_implicit_wakeup(void) {
       // if ((rte_atomic16_read(flag_p) ==1)) {
                rte_atomic16_set(flag_p, 0);
                onvm_nf_wake_notify(nf_info);
        //}
}
#endif //#ifdef INTERRUPT_SEM

static inline void start_ppkt_processing_cost(uint64_t *start_tsc) {
        if (unlikely(counter % SAMPLING_RATE == 0)) {
                *start_tsc = onvm_util_get_current_cpu_cycles();//compute_start_cycles(); //rte_rdtsc();
        }
}
static inline void end_ppkt_processing_cost(uint64_t start_tsc) {
        if (unlikely(counter % SAMPLING_RATE == 0)) {
                this_nf->stats.comp_cost = onvm_util_get_elapsed_cpu_cycles(start_tsc);
                if (likely(this_nf->stats.comp_cost > RTDSC_CYCLE_COST)) {
                        this_nf->stats.comp_cost -= RTDSC_CYCLE_COST;
                }
#ifdef STORE_HISTOGRAM_OF_NF_COMPUTATION_COST
                hist_store_v2(&nf_info->ht2, this_nf->stats.comp_cost);
                //avoid updating 'nf_info->comp_cost' as it will be calculated in the weight assignment function
                //nf_info->comp_cost  = hist_extract_v2(&nf_info->ht2,VAL_TYPE_RUNNING_AVG);
#else   //just save the running average
                nf_info->comp_cost  = (nf_info->comp_cost == 0)? (this_nf->stats.comp_cost): ((nf_info->comp_cost+this_nf->stats.comp_cost)/2);

#endif //STORE_HISTOGRAM_OF_NF_COMPUTATION_COST

#ifdef ENABLE_TIMER_BASED_NF_CYCLE_COMPUTATION
                counter = 1;
#endif  //ENABLE_TIMER_BASED_NF_CYCLE_COMPUTATION

#ifdef ENABLE_ECN_CE
                hist_store_v2(&nf_info->ht2_q, rte_ring_count(rx_ring));
#endif
        }

        #ifndef ENABLE_TIMER_BASED_NF_CYCLE_COMPUTATION
        counter++;  //computing for first packet makes also account reasonable cycles for cache-warming.
        #endif //ENABLE_TIMER_BASED_NF_CYCLE_COMPUTATION
}
#endif  //INTERRUPT_SEM
#ifdef ENABLE_NFV_RESL
static inline void
onvm_nflib_wait_till_notification(struct onvm_nf_info *nf_info) {
        //printf("\n Client [%d] is paused and waiting for SYNC Signal\n", nf_info->instance_id);
#ifdef ENABLE_LOCAL_LATENCY_PROFILER
        onvm_util_get_start_time(&ts);
#endif
        do {
                onvm_nf_yeild(nf_info,YEILD_DUE_TO_EXPLICIT_REQ);
                /* Next Check for any Messages/Notifications */
                onvm_nflib_dequeue_messages(nf_info);
        }while(((NF_PAUSED == (nf_info->status & NF_PAUSED))||(NF_WT_ND_SYNC_BIT == (nf_info->status & NF_WT_ND_SYNC_BIT))) && keep_running );

#ifdef ENABLE_LOCAL_LATENCY_PROFILER
        //printf("SIGNAL_TIME(PAUSE-->RESUME): %li ns\n", onvm_util_get_elapsed_time(&ts));
#endif
        //printf("\n Client [%d] completed wait on SYNC Signal \n", nf_info->instance_id);
}
#endif //ENABLE_NFV_RESL

static inline void onvm_nflib_check_and_wait_if_interrupted(__attribute__((unused)) struct onvm_nf_info *nf_info);
static inline void onvm_nflib_check_and_wait_if_interrupted(__attribute__((unused)) struct onvm_nf_info *nf_info) {
#if defined (INTERRUPT_SEM) && ((defined(NF_BACKPRESSURE_APPROACH_2) || defined(USE_ARBITER_NF_EXEC_PERIOD)) || defined(ENABLE_NFV_RESL))
        if(unlikely(NF_PAUSED == (nf_info->status & NF_PAUSED))) {
                //printf("\n Explicit Pause request from ONVM_MGR\n ");
                onvm_nflib_wait_till_notification(nf_info);
                //printf("\n Explicit Pause Completed by NF\n");
        }
        else if (unlikely(rte_atomic16_read(flag_p) ==1)) {
                //printf("\n Explicit Yield request from ONVM_MGR\n ");
                onvm_nf_yeild(nf_info,YEILD_DUE_TO_EXPLICIT_REQ);
                //printf("\n Explicit Yield Completed by NF\n");
        }
#endif
}

#if defined(ENABLE_SHADOW_RINGS)
static inline void onvm_nflib_handle_tx_shadow_ring(void);
static inline void onvm_nflib_handle_tx_shadow_ring(void) {

        /* Foremost Move left over processed packets from Tx shadow ring to the Tx Ring if any */
        if(unlikely( (rte_ring_count(tx_sring)))) {
                uint16_t nb_pkts = CLIENT_SHADOW_RING_SIZE;
                uint16_t tx_spkts;
                void *pkts[CLIENT_SHADOW_RING_SIZE];
                do
                {
                        // Extract packets from Tx shadow ring
                        tx_spkts = rte_ring_dequeue_burst(tx_sring, pkts, nb_pkts);

                        //fprintf(stderr, "\n Move processed packets from Shadow Tx Ring to Tx Ring [%d] packets from shadow ring( Re-queue)!\n", tx_spkts);
                        //Push the packets to the Tx ring
                        //if(unlikely(rte_ring_enqueue_bulk(tx_ring, pkts, tx_spkts) == -ENOBUFS)) { //OLD API
                        if(unlikely(rte_ring_enqueue_bulk(tx_ring, pkts, tx_spkts) == 0)) { //new API
#ifdef INTERRUPT_SEM
                                //To preserve the packets, re-enqueue packets back to the the shadow ring
                                rte_ring_enqueue_bulk(tx_sring, pkts, tx_spkts);

                                //printf("\n Yielding till Tx Ring has space for tx_shadow buffer Packets \n");
                                onvm_nf_yeild(nf_info,YIELD_DUE_TO_FULL_TX_RING);
                                //printf("\n Resuming till Tx Ring has space for tx_shadow buffer Packets \n");
#endif
                        }
                }while(rte_ring_count(tx_sring) && keep_running);
                this_nf->stats.tx += tx_spkts;
        }
}
#endif

#ifdef ENABLE_REPLICA_STATE_UPDATE
static inline void synchronize_replica_nf_state_memory(void) {

        //if(likely(nf_info->nf_state_mempool && pReplicaStateMempool))
        if(likely(dirty_state_map && dirty_state_map->dirty_index)) {
#ifdef ENABLE_LOCAL_LATENCY_PROFILER
#if 0
                static int count = 0;uint64_t start_cycle=0;
                count++;
                if(count == 1000*1000*20) {
                        onvm_util_get_start_time(&ts);
                        start_cycle = onvm_util_get_current_cpu_cycles();
                }
#endif
#endif
                //Note: Must always ensure that dirty_map is carried over first; so that the remote replica can use this value to update only the changed states
                uint64_t dirty_index = dirty_state_map->dirty_index;
                uint64_t copy_index = 0;
                uint64_t copy_setbit = 0;
                uint16_t copy_offset = 0;
                for(;dirty_index;copy_index++) {
                        copy_setbit = (1L<<(copy_index));
                        if(dirty_index&copy_setbit) {
                                copy_offset = copy_index*DIRTY_MAP_PER_CHUNK_SIZE;
                                rte_memcpy(( ((uint8_t*)pReplicaStateMempool)+copy_offset),(((uint8_t*)nf_info->nf_state_mempool)+copy_offset),DIRTY_MAP_PER_CHUNK_SIZE);
                                dirty_index^=copy_setbit;
                        }// copy_index++;
                }
                dirty_state_map->dirty_index =0;
#ifdef ENABLE_LOCAL_LATENCY_PROFILER
#if 0
                if(count == 1000*1000*20) {
                        fprintf(stdout, "STATE REPLICATION TIME (Scan + Copy): %li(ns) and %li (cycles) \n", onvm_util_get_elapsed_time(&ts), onvm_util_get_elapsed_cpu_cycles(start_cycle));
                        count=0;
                }
#endif
#endif
        }
        return;
}
#endif

#ifdef ENABLE_NFLIB_PER_FLOW_TS_STORE
//update the TS for the processed packet
static inline void update_processed_packet_ts(void **pkts, unsigned max_packets);
static inline void update_processed_packet_ts(void **pkts, unsigned max_packets) {
        if(!this_nf->per_flow_ts_info) return;
        uint16_t i, ft_index=0;
        uint64_t ts[NF_PKT_BATCH_SIZE];
        onvm_util_get_marked_packet_timestamp((struct rte_mbuf**)pkts, ts, max_packets);
        for(i=0; i< max_packets;i++) {
                struct onvm_pkt_meta *meta = onvm_get_pkt_meta((struct rte_mbuf*) pkts[i]);
#ifdef ENABLE_FT_INDEX_IN_META
                        ft_index = meta->ft_index; //(uint16_t) MAP_SDN_FT_INDEX_TO_VLAN_STATE_TBL_INDEX(meta->ft_index);
#else
                {
                        struct onvm_flow_entry *flow_entry = NULL;
                        onvm_flow_dir_get_pkt((struct rte_mbuf*) pkts[i], &flow_entry);
                        if(flow_entry) {
                                ft_index = meta->ft_index; //(uint16_t) MAP_SDN_FT_INDEX_TO_VLAN_STATE_TBL_INDEX(flow_entry->entry_index);
                        } else continue;
                }
#endif
                onvm_per_flow_ts_info_t *t_info = (onvm_per_flow_ts_info_t*)(((dirty_mon_state_map_tbl_t*)this_nf->per_flow_ts_info)+1);
                t_info[ft_index].ts = ts[i];
        }
}
#endif

static inline int onvm_nflib_fetch_packets( void **pkts, unsigned max_packets);
static inline int onvm_nflib_fetch_packets( void **pkts, unsigned max_packets) {
#if defined(ENABLE_SHADOW_RINGS)

        /* Address the buffers in the Tx Shadow Ring before starting to process the new packets */
        onvm_nflib_handle_tx_shadow_ring();

        /* First Dequeue the packets pulled from Rx Shadow Ring if not empty*/
        if (unlikely( (rte_ring_count(rx_sring)))) {
                max_packets = rte_ring_dequeue_burst(rx_sring, pkts, max_packets);
                fprintf(stderr, "Dequeued [%d] packets from shadow ring( Re-Run)!\n", max_packets);
        }
        /* ELSE: Get Packets from Main Rx Ring */
        else
#endif
        max_packets = (uint16_t)rte_ring_dequeue_burst(rx_ring, pkts, max_packets, NULL);

        if(likely(max_packets)) {
#if defined(ENABLE_SHADOW_RINGS)
                /* Also enqueue the packets pulled from Rx ring or Rx Shadow into Rx Shadow Ring */
                if (unlikely(rte_ring_enqueue_bulk(rx_sring, pkts, max_packets) == 0)) {
                        fprintf(stderr, "Enqueue: %d packets to shadow ring Failed!\n", max_packets);
                }
#endif
        } else { //if(0 == max_packets){
#ifdef INTERRUPT_SEM
                //printf("\n Yielding till Rx Ring has Packets to process \n");
                onvm_nf_yeild(nf_info,YIELD_DUE_TO_EMPTY_RX_RING);
                //printf("\n Resuming from Rx Ring has Packets to process \n");
#endif
        }
        return max_packets;
}
static
inline int onvm_nflib_post_process_packets_batch(struct onvm_nf_info *nf_info, void **pktsTX, unsigned tx_batch_size, __attribute__((unused)) unsigned non_det_evt, __attribute__((unused)) uint64_t ts_info);
static
inline int onvm_nflib_post_process_packets_batch(struct onvm_nf_info *nf_info, void **pktsTX, unsigned tx_batch_size, __attribute__((unused)) unsigned non_det_evt, __attribute__((unused)) uint64_t ts_info) {
        int ret = 0;
        /* Perform Post batch processing actions */
        /** Atomic Operations:
         * Synchronize the NF Memory State
         * Update TS of last processed packet.
         * Clear the Processed Batch of Rx packets.
         */
#ifdef ENABLE_NF_PAUSE_TILL_OUTSTANDING_NDSYNC_COMMIT
        if(unlikely(non_det_evt)){
                if(likely(nf_info->bNDSycn)) {//if(unlikely(bNDSync)) {
                        //explicitly move to pause state and wait till notified.
                        //nf_info->status = NF_PAUSED|NF_WT_ND_SYNC_BIT;
                        nf_info->status |= NF_WT_ND_SYNC_BIT;
#ifdef  __DEBUG_NDSYNC_LOGS__
                        //printf("\n Client [%d] is halting due to second ND at: [%li] while first [%li] is not committed! waiting for SYNC Signal\n", nf_info->instance_id,ts_info, nf_info->bLastPktId);
                        onvm_util_get_start_time(&nd_ts);
                        //printf("RUN-->ND_SYNC): %i ns\n", 1);
#endif
                        //wait_till_the NDSync is not signalled to be committed
                        onvm_nflib_wait_till_notification(nf_info);
#ifdef  __DEBUG_NDSYNC_LOGS__
                       // printf("\n\n\n\n\n\n\n\n$$$$$$$$$$$$WAIT_TIME(ND_SYNC): %li ns $$$$$$$$$$$$$\n\n\n\n\n\n\n", (delta_nd=onvm_util_get_elapsed_time(&nd_ts)));
                        if(min_nd==0 || delta_nd < min_nd) min_nd= delta_nd;
                        if(delta_nd > max_nd)max_nd=delta_nd;
                        if(avg_nd) avg_nd = (avg_nd+delta_nd)/2;
                        else avg_nd= delta_nd;
                        //printf("\n\n In wait_till_Notitfication: WAIT_TIME_STATS(ND_SYNC):\n Cur=%li\n Min= %li\n Max: %li \n Avg: %li \n", delta_nd, min_nd, max_nd, avg_nd);
                        nf_info->min_nd=min_nd; nf_info->max_nd=max_nd; nf_info->avg_nd=avg_nd;
#endif
                } else {
#ifdef  __DEBUG_NDSYNC_LOGS__
                        //printf("\n Client [%d] got first ND SYNC event at: %li! \n", nf_info->instance_id, ts_info);
                        onvm_util_get_start_time(&nd_ts);
#endif
                }
                nf_info->bNDSycn=1; //bNDSync = 1;    //set the NDSync to True again
                nf_info->bLastPktId=ts_info;    //can be used to check if Resume message carries TxTs of latest synced packet
        }
        //printf("\n %d", non_det_evt);
#endif
#ifdef REPLICA_STATE_UPDATE_MODE_PER_BATCH
        synchronize_replica_nf_state_memory();
#endif
#ifdef PER_FLOW_TS_UPDATE_PER_BATCH
        //update the TS for the processed batch of packets
        update_processed_packet_ts(pktsTX,tx_batch_size);
#endif

#if defined(SHADOW_RING_UPDATE_PER_BATCH)
        //rte_ring_enqueue_bulk(tx_sring, pktsTX, tx_batch_size);
        //void *pktsRX[NF_PKT_BATCH_SIZE];
        //rte_ring_sc_dequeue_bulk(rx_sring, pktsRX,NF_PKT_BATCH_SIZE ); //for now Bypass as we do at the end (down)
#endif

#if defined(TEST_MEMCPY_MODE_PER_BATCH)
        do_memcopy(nf_info->nf_state_mempool);
#endif //TEST_MEMCPY_OVERHEAD

#ifdef MIMIC_FTMB
        generate_and_transmit_pals_for_batch(pktsTX, tx_batch_size, non_det_evt,ts_info);
#endif
        //if(likely(tx_batch_size)) {
        //if(likely(-ENOBUFS != (ret = rte_ring_enqueue_bulk(tx_ring, pktsTX, tx_batch_size, NULL)))) {
        if(likely(0 != (ret = rte_ring_enqueue_bulk(tx_ring, pktsTX, tx_batch_size, NULL)))) {
                this_nf->stats.tx += tx_batch_size;
        } else {
#if defined(NF_LOCAL_BACKPRESSURE)
                do {
#ifdef INTERRUPT_SEM
                        //printf("\n Yielding till Tx Ring has place to store Packets\n");
                        onvm_nf_yeild(nf_info, YIELD_DUE_TO_FULL_TX_RING);
                        //printf("\n Resuming from Tx Ring wait to store Packets\n");
#endif
                        if (tx_batch_size > rte_ring_free_count(tx_ring)) {
                                continue;
                        }
                        //if((ret = rte_ring_enqueue_bulk(tx_ring,pktsTX,tx_batch_size,NULL)) != -ENOBUFS){ ret = 0; break;}
                        if((ret = rte_ring_enqueue_bulk(tx_ring,pktsTX,tx_batch_size,NULL)) != 0){ ret = 0; break;}
                } while (ret && ((this_nf->info->status==NF_RUNNING) && keep_running));
                this_nf->stats.tx += tx_batch_size;
#endif  //NF_LOCAL_BACKPRESSURE
                }
#if defined(ENABLE_SHADOW_RINGS)
        /* Finally clear all packets from the Tx Shadow Ring and also Rx shadow Ring ::only if packets from shadow ring have been flushed to Tx Ring: Reason, NF might get paused or stopped */
        if(likely(ret == 0)) {
                rte_ring_sc_dequeue_burst(tx_sring,pktsTX,rte_ring_count(tx_sring));
                if(unlikely(rte_ring_count(rx_sring))) {
                        //These are the held packets in the NF in this round:
                        rte_ring_sc_dequeue_burst(rx_sring,pktsTX,rte_ring_count(rx_sring));
                        //fprintf(stderr, "BATCH END: %d packets still in Rx shadow ring!\n", rte_ring_sc_dequeue_burst(rx_sring,pkts,rte_ring_count(rx_sring)));
                }
        }
#endif
        //}
        return ret;
}
static inline int onvm_nflib_process_packets_batch(struct onvm_nf_info *nf_info, void **pkts, unsigned nb_pkts,  pkt_handler_func  handler);
static inline int onvm_nflib_process_packets_batch(struct onvm_nf_info *nf_info, void **pkts, unsigned nb_pkts,  pkt_handler_func  handler) {
        int ret=0;
        uint16_t i=0;
        uint16_t tx_batch_size = 0;
        void *pktsTX[NF_PKT_BATCH_SIZE];
        uint8_t bCurND=0;
        uint64_t bCurNDPktId=0;
#ifdef ONVM_GPU
	void *packet_data;
#endif
	
#ifdef INTERRUPT_SEM
        // To account NFs computation cost (sampled over SAMPLING_RATE packets)
        uint64_t start_tsc = 0;
#endif
        for (i = 0; i < nb_pkts; i++) {

#ifdef INTERRUPT_SEM
                start_ppkt_processing_cost(&start_tsc);
#endif

#ifdef ONVM_GPU
		//we shall copy the packets here itself.. why should we give it to the handler
		if(onvm_pkt_ipv4_hdr(pkts[i]) != NULL){
		  packet_data = rte_pktmbuf_mtod_offset((struct rte_mbuf *)pkts[i], void *, sizeof(struct ether_hdr)+sizeof(struct ipv4_hdr)+sizeof(struct udp_hdr));
		  copy_data_to_image(packet_data, nf_info);
		}
#endif
                ret = (*handler)((struct rte_mbuf*) pkts[i], onvm_get_pkt_meta((struct rte_mbuf*) pkts[i]), nf_info);

#ifdef ENABLE_NF_PAUSE_TILL_OUTSTANDING_NDSYNC_COMMIT
                if(unlikely(onvm_get_pkt_meta((struct rte_mbuf*) pkts[i])->reserved_word&NF_NEED_ND_SYNC)) {
                        bCurND=1;
                        bCurNDPktId = ((struct rte_mbuf*)pkts[i])->ol_flags;
                }
#endif

#if defined(TEST_MEMCPY_MODE_PER_PACKET)
                do_memcopy(nf_info->nf_state_mempool);
#endif

#ifdef INTERRUPT_SEM
                end_ppkt_processing_cost(start_tsc);
#endif  //INTERRUPT_SEM

                /* NF returns 0 to return packets or 1 to buffer */
                if (likely(ret == 0)) {
                        pktsTX[tx_batch_size++] = pkts[i];

#ifdef REPLICA_STATE_UPDATE_MODE_PER_PACKET
                synchronize_replica_nf_state_memory();
#endif
#ifdef PER_FLOW_TS_UPDATE_PER_PKT
                //update the TS for the processed packet
                update_processed_packet_ts(&pkts[i],1);
#endif
#if defined(SHADOW_RING_UPDATE_PER_PKT)
                        /* Move this processed packet (Head of Rx shadow Ring) to Tx Shadow Ring */
                        void *pkt_rx;
                        rte_ring_sc_dequeue(rx_sring, &pkt_rx);
                        rte_ring_sp_enqueue(tx_sring, pkts[i]);
#endif
                }
                else {
#ifdef ENABLE_NF_TX_STAT_LOGS
                        this_nf->stats.tx_buffer++;
#endif
#if defined(SHADOW_RING_UPDATE_PER_PKT)
                        /* Remove this buffered packet from Rx shadow Ring, Should we buffer it separately, or assume NF has held on to it and NF state update reflects it. */
                        void *pkt_rx;
                        rte_ring_sc_dequeue(rx_sring, &pkt_rx);
                        //rte_ring_sp_enqueue(rx_sring, pkts[i]); //TODO: Need separate buffer packets holder; cannot use the rx_sring
#endif
                }
        } //End Batch Process;

#if defined(SHADOW_RING_UPDATE_PER_BATCH)
        void *pktsRX[NF_PKT_BATCH_SIZE];
        //clear rx_sring
        rte_ring_sc_dequeue_bulk(rx_sring, pktsRX,NF_PKT_BATCH_SIZE);
        //save processed packets in tx_sring
        rte_ring_enqueue_bulk(tx_sring, pktsTX, tx_batch_size);
#endif

        /* Perform Post batch processing actions */
        if(likely(tx_batch_size)) {
                return onvm_nflib_post_process_packets_batch(nf_info, pktsTX, tx_batch_size, bCurND, bCurNDPktId);
        }
        return ret;
}

int
onvm_nflib_run_callback(struct onvm_nf_info* nf_info, pkt_handler_func handler, callback_handler_func callback) {
        void *pkts[NF_PKT_BATCH_SIZE]; //better to use (NF_PKT_BATCH_SIZE*2)
        uint16_t nb_pkts;

        pkt_hdl_func = handler;
        printf("\nClient process %d handling packets\n", nf_info->instance_id);
        printf("[Press Ctrl-C to quit ...]\n");

        /* Listen for ^C so we can exit gracefully */
        signal(SIGINT, onvm_nflib_handle_signal);
        
        onvm_nflib_notify_ready(nf_info);

	printf("------ ***** ------- ##### We notified NFLIB ----- ***** ------\n");
	
        /* First Check for any Messages/Notifications */
        onvm_nflib_dequeue_messages(nf_info);

#ifdef ENABLE_LOCAL_LATENCY_PROFILER
        printf("WAIT_TIME(INIT-->START-->RUN-->RUNNING): %li ns\n", onvm_util_get_elapsed_time(&g_ts));
#endif

        for (;keep_running;) {
                /* check if signaled to block, then block:: TODO: Merge this to the Message above */
                onvm_nflib_check_and_wait_if_interrupted(nf_info);

		//printf("------ ***** ------- ##### We got inside keep_running after wait if interrupted ----- ***** ------\n");

                nb_pkts = onvm_nflib_fetch_packets(pkts, NF_PKT_BATCH_SIZE);
                if(likely(nb_pkts)) {
                        /* Give each packet to the user processing function */
                        nb_pkts = onvm_nflib_process_packets_batch(nf_info, pkts, nb_pkts, handler);
#ifdef ONVM_GPU
			//add the counters to number of packets that are outstanding...
			/* Since we know how many packets make images for us now.. so we can just determine how many images 
			 *are left here */
#ifdef ONVM_GPU_SAME_SIZE_PKTS
			nf_info->number_of_pkts_outstanding += nb_pkts; //added the number of packets
#endif
#endif
                }

#ifdef ENABLE_TIMER_BASED_NF_CYCLE_COMPUTATION
                rte_timer_manage();
#endif  //ENABLE_TIMER_BASED_NF_CYCLE_COMPUTATION

                /* Finally Check for any Messages/Notifications */
                onvm_nflib_dequeue_messages(nf_info);
		//printf("------ ***** ------- ##### We got after nflib_dequeue if interrupted ----- ***** ------\n");

                if(callback) {
                        keep_running = !(*callback)(nf_info) && keep_running;
                }
        }

        printf("\n NF is Exiting...!\n");
        onvm_nflib_cleanup(nf_info);
        return 0;
}

int
onvm_nflib_run(struct onvm_nf_info* nf_info, pkt_handler_func handler) {
        return onvm_nflib_run_callback(nf_info, handler, ONVM_NO_CALLBACK);
}


int
onvm_nflib_return_pkt(struct onvm_nf_info* nf_info,  struct rte_mbuf* pkt) {
        return onvm_nflib_return_pkt_bulk(nf_info, &pkt, 1);
		/* FIXME: should we get a batch of buffered packets and then enqueue? Can we keep stats? */
        if(unlikely(rte_ring_enqueue(tx_ring, pkt) == -ENOBUFS)) {
                rte_pktmbuf_free(pkt);
                this_nf->stats.tx_drop++;
                return -ENOBUFS;
        }
        else {
#ifdef ENABLE_NF_TX_STAT_LOGS
                this_nf->stats.tx_returned++;
#endif
        }
        return 0;
}

int
onvm_nflib_return_pkt_bulk(struct onvm_nf_info *nf_info, struct rte_mbuf** pkts, uint16_t count)  {
        unsigned int i;
        if (pkts == NULL || count == 0) return -1;
        if (unlikely(rte_ring_enqueue_bulk(nfs[nf_info->instance_id].tx_q, (void **)pkts, count, NULL) == 0)) {
                nfs[nf_info->instance_id].stats.tx_drop += count;
                for (i = 0; i < count; i++) {
                        rte_pktmbuf_free(pkts[i]);
                }
                return -ENOBUFS;
        }
        else nfs[nf_info->instance_id].stats.tx_returned += count;
        return 0;
}

void
onvm_nflib_stop(__attribute__((unused)) struct onvm_nf_info* nf_info) {
        rte_exit(EXIT_SUCCESS, "Done.");
}

int
onvm_nflib_init(int argc, char *argv[], const char *nf_tag, struct onvm_nf_info **nf_info_p) {
        int retval_parse, retval_final;
		struct onvm_nf_info *nf_info;
		int retval_eal = 0;
        //int use_config = 0;
		const struct rte_memzone *mz_nf;
        const struct rte_memzone *mz_port;
        const struct rte_memzone *mz_scp;
        const struct rte_memzone *mz_services;
        const struct rte_memzone *mz_nf_per_service;
        //struct rte_mempool *mp;
        struct onvm_service_chain **scp;

#ifdef ENABLE_LOCAL_LATENCY_PROFILER
        onvm_util_get_start_time(&g_ts);
#endif
        if ((retval_eal = rte_eal_init(argc, argv)) < 0)
                return -1;

        /* Modify argc and argv to conform to getopt rules for parse_nflib_args */
        argc -= retval_eal; argv += retval_eal;

        /* Reset getopt global variables opterr and optind to their default values */
        opterr = 0; optind = 1;

        if ((retval_parse = onvm_nflib_parse_args(argc, argv)) < 0)
                rte_exit(EXIT_FAILURE, "Invalid command-line arguments\n");

        /*
         * Calculate the offset that the nf will use to modify argc and argv for its
         * getopt call. This is the sum of the number of arguments parsed by
         * rte_eal_init and parse_nflib_args. This will be decremented by 1 to assure
         * getopt is looking at the correct index since optind is incremented by 1 each
         * time "--" is parsed.
         * This is the value that will be returned if initialization succeeds.
         */
        retval_final = (retval_eal + retval_parse) - 1;

        /* Reset getopt global variables opterr and optind to their default values */
        opterr = 0; optind = 1;

        /* Lookup mempool for nf_info struct */
        nf_info_mp = rte_mempool_lookup(_NF_MEMPOOL_NAME);
        if (nf_info_mp == NULL)
                rte_exit(EXIT_FAILURE, "No Client Info mempool - bye\n");

        /* Lookup mempool for NF messages */
        nf_msg_pool = rte_mempool_lookup(_NF_MSG_POOL_NAME);
        if (nf_msg_pool == NULL)
                rte_exit(EXIT_FAILURE, "No NF Message mempool - bye\n");

#ifdef ONVM_GPU
	/* Mempool for images */
	nf_image_pool = rte_mempool_lookup(_NF_IMAGE_POOL_NAME);
	if(nf_image_pool == NULL)
	  rte_exit(EXIT_FAILURE, "No Image Mempool -bye\n");
#endif
        /* Initialize the info struct */
        nf_info = onvm_nflib_info_init(nf_tag);
        *nf_info_p = nf_info;
		
        pktmbuf_pool = rte_mempool_lookup(PKTMBUF_POOL_NAME);
        if (pktmbuf_pool == NULL)
                rte_exit(EXIT_FAILURE, "Cannot get mempool for mbufs\n");

        /* Lookup mempool for NF structs */
        mz_nf = rte_memzone_lookup(MZ_NF_INFO);
        if (mz_nf == NULL)
                rte_exit(EXIT_FAILURE, "Cannot get NF structure mempool\n");
        nfs = mz_nf->addr;

        mz_services = rte_memzone_lookup(MZ_SERVICES_INFO);
        if (mz_services == NULL) {
                rte_exit(EXIT_FAILURE, "Cannot get service information\n");
        }
        services = mz_services->addr;

        mz_nf_per_service = rte_memzone_lookup(MZ_NF_PER_SERVICE_INFO);
        if (mz_nf_per_service == NULL) {
                rte_exit(EXIT_FAILURE, "Cannot get NF per service information\n");
        }
        nf_per_service_count = mz_nf_per_service->addr;

        mz_port = rte_memzone_lookup(MZ_PORT_INFO);
        if (mz_port == NULL)
                rte_exit(EXIT_FAILURE, "Cannot get port info structure\n");
        ports = mz_port->addr;

        mz_scp = rte_memzone_lookup(MZ_SCP_INFO);
        if (mz_scp == NULL)
                rte_exit(EXIT_FAILURE, "Cannot get service chain info structre\n");
        scp = mz_scp->addr;
        default_chain = *scp;

        onvm_sc_print(default_chain);

        mgr_msg_ring = rte_ring_lookup(_MGR_MSG_QUEUE_NAME);
        if (mgr_msg_ring == NULL)
                rte_exit(EXIT_FAILURE, "Cannot get MGR Message ring");
#ifdef ENABLE_SYNC_MGR_TO_NF_MSG
        mgr_rsp_ring = rte_ring_lookup(_MGR_RSP_QUEUE_NAME);
        if (mgr_rsp_ring == NULL)
                rte_exit(EXIT_FAILURE, "Cannot get MGR response (SYNC) ring");
#endif
        onvm_nflib_start_nf(nf_info);

#ifdef INTERRUPT_SEM
        init_shared_cpu_info(nf_info->instance_id);
#endif

#ifdef USE_CGROUPS_PER_NF_INSTANCE
        init_cgroup_info(nf_info);
#endif

#ifdef ENABLE_TIMER_BASED_NF_CYCLE_COMPUTATION
        init_nflib_timers();
#endif  //ENABLE_TIMER_BASED_NF_CYCLE_COMPUTATION

#ifdef STORE_HISTOGRAM_OF_NF_COMPUTATION_COST
        hist_init_v2(&nf_info->ht2);    //hist_init( &ht, MAX_NF_COMP_COST_CYCLES);
#endif

#ifdef ENABLE_ECN_CE
        hist_init_v2(&nf_info->ht2_q);    //hist_init( &ht, MAX_NF_COMP_COST_CYCLES);
#endif

#ifdef TEST_MEMCPY_OVERHEAD
        allocate_base_memory();
#endif

#ifdef ENABLE_LOCAL_LATENCY_PROFILER
        int64_t ttl_elapsed = onvm_util_get_elapsed_time(&g_ts);
        printf("WAIT_TIME(INIT-->START-->Init_end): %li ns\n", ttl_elapsed);
#endif
        // Get the FlowTable Entries Exported to the NF.
        onvm_flow_dir_nf_init();

        RTE_LOG(INFO, APP, "Finished Process Init.\n");
        return retval_final;
}

int
onvm_nflib_drop_pkt(struct rte_mbuf* pkt) {
        rte_pktmbuf_free(pkt);
        this_nf->stats.tx_drop++;
        return 0;
}

void notify_for_ecb(void) {
        need_ecb = 1;
#ifdef INTERRUPT_SEM
        if ((rte_atomic16_read(flag_p) ==1)) {
            onvm_nf_wake_notify(nf_info);
        }
#endif
        return;
}

int
onvm_nflib_handle_msg(struct onvm_nf_msg *msg, __attribute__((unused)) struct onvm_nf_info *nf_info) {
        switch(msg->msg_type) {
        printf("\n Received MESSAGE [%d]\n!!", msg->msg_type);
        case MSG_STOP:
                keep_running = 0;
                if(NF_PAUSED == (nf_info->status & NF_PAUSED)) {
                        nf_info->status = NF_RUNNING;
#ifdef INTERRUPT_SEM
                        onvm_nflib_implicit_wakeup(); //TODO: change this ecb call; split ecb call to two funcs. sounds stupid but necessary as cache update of flag_p takes time; otherwise results in sleep-wkup cycles
#endif
                }
                RTE_LOG(INFO, APP, "Shutting down...\n");
                break;
        case MSG_NF_TRIGGER_ECB:
                notify_for_ecb();
                break;

		/* Aditya's edit ONVM_GPU*/
#ifdef ONVM_GPU
	case MSG_GPU_MODEL:
	       if((*nf_gpu_func)(msg))
		 printf("The NF didn't process GPU_MODEL message well \n");
	       break;
	case MSG_GET_GPU_READY:
	       if((*nf_gpu_func)(msg))
		 printf("The NF didn't process GET_GPU_READY message well \n");
	       break;
	case MSG_RESTART:
	       if((*nf_gpu_func)(msg))
		 printf("The NF didn't process MSG_RESTART message well \n");
	       break;
#endif
	  /* Aditya's edit end */
	  
        case MSG_PAUSE:
                if(NF_PAUSED != (nf_info->status & NF_PAUSED)) {
                        RTE_LOG(INFO, APP, "NF Status changed to Pause!...\n");
                        nf_info->status |= NF_PAUSED;   //change == to |=
                }
                RTE_LOG(INFO, APP, "NF Pausing!...\n");
                break;
        case MSG_RESUME: //MSG_RUN
#ifdef ENABLE_NF_PAUSE_TILL_OUTSTANDING_NDSYNC_COMMIT
                if(likely(nf_info->bNDSycn)) { //if(unlikely(bNDSync)) {
                        nf_info->bNDSycn=0; //bNDSync=0;
                        //printf("\n Received NDSYNC_CLEAR AND RESUME MESSAGE\n!!");
#ifdef  __DEBUG_NDSYNC_LOGS__
                        delta_nd=onvm_util_get_elapsed_time(&nd_ts);
                        if(min_nd==0 || delta_nd < min_nd) min_nd= delta_nd;
                        if(delta_nd > max_nd)max_nd=delta_nd;
                        if(avg_nd) avg_nd = (avg_nd+delta_nd)/2;
                        else avg_nd= delta_nd;
                        //printf("\n\n IN RESUME NOTIFICATION:: WAIT_TIME_STATS(ND_SYNC):\n Cur=%li\n Min= %li\n Max: %li \n Avg: %li \n", delta_nd, min_nd, max_nd, avg_nd);
                        nf_info->min_nd=min_nd; nf_info->max_nd=max_nd; nf_info->avg_nd=avg_nd;
#endif
                }
#endif
                nf_info->status = NF_RUNNING;   //set = RUNNING; works for both NF_PAUSED and NF_WT_ND_SYNC_BIT cases!
#ifdef INTERRUPT_SEM
                        onvm_nflib_implicit_wakeup(); //TODO: change this ecb call; split ecb call to two funcs. sounds stupid but necessary as cache update of flag_p takes time; otherwise results in sleep-wkup cycles
#endif

                RTE_LOG(DEBUG, APP, "Resuming NF...\n");
                break;
        case MSG_NOOP:
        default:
                break;
        }
        return 0;
}

static inline void
onvm_nflib_dequeue_messages(struct onvm_nf_info *nf_info) {
        // Check and see if this NF has any messages from the manager
        if (likely(rte_ring_count(nf_msg_ring) == 0)) {
                return;
        }
        struct onvm_nf_msg *msg = NULL;
        rte_ring_dequeue(nf_msg_ring, (void**)(&msg));
        onvm_nflib_handle_msg(msg, nf_info);
        rte_mempool_put(nf_msg_pool, (void*)msg);
}
/******************************Helper functions*******************************/
static struct onvm_nf_info *
onvm_nflib_info_init(const char *tag)
{
        void *mempool_data;
        struct onvm_nf_info *info;

        if (rte_mempool_get(nf_info_mp, &mempool_data) < 0) {
                rte_exit(EXIT_FAILURE, "Failed to get client info memory");
        }

        if (mempool_data == NULL) {
                rte_exit(EXIT_FAILURE, "Client Info struct not allocated");
        }

        info = (struct onvm_nf_info*) mempool_data;
        info->instance_id = initial_instance_id;
        info->service_id = service_id;
        info->status = NF_WAITING_FOR_ID;
        info->tag = tag;

        info->pid = getpid();
	
#ifdef ONVM_GPU
	//initialize the array that is a queue for GPU
	gpu_queue_image_id = (int *) rte_malloc(NULL, sizeof(int )*MAX_IMAGE, 0);
	//array for noting what time the image is inserted in GPU queue
	gpu_callbacks = (struct gpu_callback *) rte_malloc(NULL, sizeof(struct gpu_callback)*MAX_IMAGE,0);
	
	int original_instance_id = initial_instance_id;
	if(is_secondary_active_nf_id(info->instance_id)){
	  original_instance_id = get_associated_active_or_standby_nf_id(info->instance_id);
	  printf("DEBUG... this is a secondary NF ---+++_---+++___ We get associated NF id %d \n",original_instance_id);
	}
	info->image_info = &all_images_information[original_instance_id];
	//initialize the histograms
	hist_init_v2(&(info->image_queueing_rate));
	hist_init_v2(&(info->image_processing_rate));
	hist_init_v2(&(info->end_to_end_image_processing_time));
	initialize_ml_timers(info);
	
#endif
        return info;
}


static void
onvm_nflib_usage(const char *progname) {
        printf("Usage: %s [EAL args] -- "
#ifdef ENABLE_STATIC_ID
               "[-n <instance_id>]"
#endif
               "[-r <service_id>]\n\n", progname);
}


static int
onvm_nflib_parse_args(int argc, char *argv[]) {
        const char *progname = argv[0];
        int c;

        opterr = 0;
#ifdef ENABLE_STATIC_ID
        while ((c = getopt (argc, argv, "n:r:")) != -1)
#else
        while ((c = getopt (argc, argv, "r:")) != -1)
#endif
                switch (c) {
#ifdef ENABLE_STATIC_ID
                case 'n':
                        initial_instance_id = (uint16_t) strtoul(optarg, NULL, 10);
                        break;
#endif
                case 'r':
                        service_id = (uint16_t) strtoul(optarg, NULL, 10);
                        // Service id 0 is reserved
                        if (service_id == 0) service_id = -1;
                        break;
                case '?':
                        onvm_nflib_usage(progname);
                        if (optopt == 'n')
                                fprintf(stderr, "Option -%c requires an argument.\n", optopt);
                        else if (isprint(optopt))
                                fprintf(stderr, "Unknown option `-%c'.\n", optopt);
                        else
                                fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
                        return -1;
                default:
                        return -1;
                }

        if (service_id == (uint16_t)-1) {
                /* Service ID is required */
                fprintf(stderr, "You must provide a nonzero service ID with -r\n");
                return -1;
        }
        return optind;
}


static void
onvm_nflib_handle_signal(int sig)
{
        if (sig == SIGINT) {
                keep_running = 0;
#ifdef INTERRUPT_SEM
                onvm_nflib_implicit_wakeup();
#endif
        }
        /* TODO: Main thread for INTERRUPT_SEM case: Must additionally relinquish SEM, SHM */
}

static inline void
onvm_nflib_cleanup(__attribute__((unused)) struct onvm_nf_info *nf_info)
{
        struct onvm_nf_msg *shutdown_msg;
        nf_info->status = NF_STOPPED;

#ifndef ENABLE_MSG_CONSTRUCT_NF_INFO_NOTIFICATION
        /* Put this NF's info struct back into queue for manager to ack shutdown */
        if (mgr_msg_ring == NULL) {
                rte_mempool_put(nf_info_mp, nf_info); // give back memory
                rte_exit(EXIT_FAILURE, "Cannot get nf_info ring for shutdown");
        }

        if (rte_ring_enqueue(mgr_msg_ring, nf_info) < 0) {
                rte_mempool_put(nf_info_mp, nf_info); // give back mermory
                rte_exit(EXIT_FAILURE, "Cannot send nf_info to manager for shutdown");
        }
        return ;
#else
        /* Put this NF's info struct back into queue for manager to ack shutdown */
        if (mgr_msg_ring == NULL) {
                rte_mempool_put(nf_info_mp, nf_info); // give back mermory
                rte_exit(EXIT_FAILURE, "Cannot get nf_info ring for shutdown");
        }
        if (rte_mempool_get(nf_msg_pool, (void**)(&shutdown_msg)) != 0) {
                rte_mempool_put(nf_info_mp, nf_info); // give back mermory
                rte_exit(EXIT_FAILURE, "Cannot create shutdown msg");
        }

        shutdown_msg->msg_type = MSG_NF_STOPPING;
        shutdown_msg->msg_data = nf_info;

        if (rte_ring_enqueue(mgr_msg_ring, shutdown_msg) < 0) {
                rte_mempool_put(nf_info_mp, nf_info); // give back mermory
                rte_mempool_put(nf_msg_pool, shutdown_msg);
                rte_exit(EXIT_FAILURE, "Cannot send nf_info to manager for shutdown");
        }
        return;
#endif
}

static inline int
onvm_nflib_notify_ready(struct onvm_nf_info *nf_info) {
        int ret = 0;

#ifdef ENABLE_LOCAL_LATENCY_PROFILER
        onvm_util_get_start_time(&ts);
#endif
        nf_info->status = NF_WAITING_FOR_RUN;

#ifdef ENABLE_MSG_CONSTRUCT_NF_INFO_NOTIFICATION
        struct onvm_nf_msg *startup_msg;
        /* Put this NF's info struct onto queue for manager to process startup */
        ret = rte_mempool_get(nf_msg_pool, (void**)(&startup_msg));
        if (ret != 0) return ret;

        startup_msg->msg_type = MSG_NF_READY;
        startup_msg->msg_data = nf_info;
        ret = rte_ring_enqueue(mgr_msg_ring, startup_msg);
        if (ret < 0) {
                rte_mempool_put(nf_msg_pool, startup_msg);
                return ret;
        }
#else
        /* Put this NF's info struct onto queue for manager to process startup */
        ret = rte_ring_enqueue(mgr_msg_ring, nf_info);
        if (ret < 0) {
                rte_mempool_put(nf_info_mp, nf_info); // give back memory
                rte_exit(EXIT_FAILURE, "Cannot send nf_info to manager");
        }
#endif
        /* Wait for a client id to be assigned by the manager */
        RTE_LOG(INFO, APP, "Waiting for manager to put to RUN state...\n");
        struct timespec req = {0,1000}, res = {0,0};
        for (; nf_info->status == (uint16_t)NF_WAITING_FOR_RUN ;) {
                nanosleep(&req, &res); //sleep(1); //better poll for some time and exit if failed within that time.?
        }
#ifdef ENABLE_LOCAL_LATENCY_PROFILER
        int64_t ttl_elapsed = onvm_util_get_elapsed_time(&ts);
        printf("WAIT_TIME(START-->RUN): %li ns\n", ttl_elapsed);
#endif

#if 0
        if(NF_PAUSED == nf_info->status) {
                onvm_nflib_wait_till_notification(nf_info);
        }
        if(NF_RUNNING != nf_info->status) {
                switch(nf_info->status) {
                case NF_PAUSED:
                        onvm_nflib_wait_till_notification(nf_info);
                        break;
                case NF_STOPPED:
                        onvm_nflib_cleanup(nf_info);
                        rte_exit(EXIT_FAILURE, "NF RUNfailed! moving to shutdown!");
                        break;
                default:
                        break;
                }
        }
#endif
        return 0;
}

/* aditya's GPU related helper functions... */
#ifdef ONVM_GPU

/* we require these functions so we can keep track of how many images are present now */
static image_data* get_image(int pkt_file_index, image_data **pending_img);
static void delete_image(image_data * image, struct onvm_nf_info * nf_info);
//GPU callback function to report after the evaluation is finished...
void CUDART_CB gpu_image_callback_function(cudaStream_t event, cudaError_t status, void *data);
//this function will be called to evaluate an image from mempool
void evaluate_an_image_from_mempool(struct onvm_nf_info * nf_info);

//ml stats function
void compute_ml_stats(struct onvm_nf_info *nf_info);

/* this function sends message to onvm manager */
void onvm_send_gpu_msg_to_mgr(provide_gpu_model *message_to_manager, int message_type){
  printf("ONVM SEND GPU MSG TO MGR\n");
  struct onvm_nf_msg * msg_mgr;
  int ret = 0;
  ret = rte_mempool_get(nf_msg_pool,(void **) &msg_mgr);
  msg_mgr->msg_type = message_type;
  msg_mgr->msg_data = message_to_manager;
  struct rte_ring * mgr_ring;
  mgr_ring = rte_ring_lookup(_MGR_MSG_QUEUE_NAME);
  if(mgr_ring == NULL)
    printf("message ring couldn't be found \n");
  ret = rte_ring_enqueue(mgr_ring, msg_mgr);
  if (ret < 0) {
    rte_mempool_put(nf_msg_pool, msg_mgr);
    return;
  }
}

/* common function to load ml file */
void load_ml_file(char * file_path, int cpu_gpu_flag, void ** cpu_func_ptr, void ** gpu_func_ptr){
  /* convert the filename to wchar_t */
  size_t filename_length = strlen(file_path);
  wchar_t file_name[filename_length];
  size_t wfilename_length = mbstowcs(file_name,file_path,filename_length+1);
  
  if(wfilename_length == 0)
    fprintf(stdout, "Something went wrong with converting filename to wide letter \n");
  
  char device[5];
  if(cpu_gpu_flag == 0)
    strcpy(device, "CPU");
  else if(cpu_gpu_flag == 1)
    strcpy(device, "GPU");

  fprintf(stdout, "Loading ML file on %s device \n", device);

  int ret = load_model(file_name, cpu_func_ptr, gpu_func_ptr, cpu_gpu_flag, 0);

  if(ret)
    fprintf(stdout, "ML File loading failure \n");
}

//this function will evaluate your data...
void evaluate_the_image(void *function_ptr, void * input_buffer, float *stats, float *output){
  //printf("%p, %p, %p, %p \n",function_ptr, input_buffer, stats, output);
  evaluate_in_gpu_input_from_host((float *)input_buffer, IMAGE_SIZE*IMAGE_BATCH, output, function_ptr, stats, 0);
  return;
}

//a more comprehensive evaluate function that can be called by the timer thread
void evaluate_an_image_from_mempool(struct onvm_nf_info * nf_info){
  //get the next image. only if there is any images to execute
  if(nf_info->image_info->num_of_ready_images <= 0){
    return;
  }

  //otherwise grab one image and proceed
  image_data *ready_image = nf_info->image_info->ready_images[nf_info->image_info->index_of_ready_image];
  
  //increase the index of ready image... as well as decrease the number of ready images
  nf_info->image_info->num_of_ready_images--;
  nf_info->image_info->index_of_ready_image++;
  nf_info->image_info->index_of_ready_image %= MAX_IMAGE; 
  
  struct gpu_callback * callback_data = &(gpu_callbacks[gpu_queue_current_index]);
  callback_data->nf_info = nf_info;
  callback_data->ready_image = ready_image;
  
  //get the time
  struct timespec eval_begin_time;
  
  //submit this image to evaluation
  //evaluate_in_gpu_input_from_host(ready_image->image_data_arr, IMAGE_SIZE*(ready_image->num_data_points_stored/IMAGE_NUM_ELE),ready_image->output,nf_info->function_ptr,ready_image->stats, gpu_finish_work_flag, &gpu_image_callback_function, (void *) ready_image);
  
  //new evaluation function, older one is too cubersome
  evaluate_image_in_gpu(ready_image, &gpu_image_callback_function, (void *) &callback_data, gpu_finish_work_flag);
  //post evaluation...put it in the GPU eval queue.
    
  gpu_queue_image_id[gpu_queue_current_index]=ready_image->image_id;
  clock_gettime(CLOCK_MONOTONIC, &eval_begin_time); //read the current time
  ready_image->timestamps[1] = eval_begin_time; //the index for timestamp is explained in onvm_image.h 
  gpu_queue_current_index++;
  gpu_queue_current_index %= MAX_IMAGE;
  num_elements_in_gpu_queue++;  
}

//GPU callback function to report after the evaluation is finished...
void CUDART_CB gpu_image_callback_function(__attribute__((unused)) cudaStream_t event, __attribute__((unused)) cudaError_t status, void *data){

  //just update the stats here for now...
  struct timespec call_back_time;
  clock_gettime(CLOCK_MONOTONIC, &call_back_time);
  struct gpu_callback *callback_data = (struct gpu_callback *) data;
  callback_data->ready_image->timestamps[3] = call_back_time;
  //get the stats updated and then delete the image
  callback_data->nf_info->number_of_images_processed++;
  //decrease the number of packets the image had
  #ifdef ONVM_GPU_SAME_SIZE_PKTS
  callback_data->nf_info->number_of_pkts_outstanding -= NUM_IN_PKTS;
  #endif

  //put the time in histogram
  hist_store_v2(&(callback_data->nf_info->end_to_end_image_processing_time),(uint32_t)time_difference_usec(&(callback_data->ready_image->timestamps[0]),&(callback_data->ready_image->timestamps[4])));
  //store the time of execution only
  hist_store_v2(&(callback_data->nf_info->end_to_end_image_processing_time),(uint32_t)time_difference_usec(&(callback_data->ready_image->timestamps[2]),&(callback_data->ready_image->timestamps[4])));

  //we can delete the image now
  delete_image(callback_data->ready_image, callback_data->nf_info);

  //reduce the number of elements in gpu queue
  num_elements_in_gpu_queue--;
  
 
}


/* initializes the images... 
 *
 * THIS FUNCTIONALITY HAS BEEN MOVED TO MANAGER
 * onvm_init.c
 *
void image_init(struct onvm_nf_info *nf, struct onvm_nf_info *original_nf){

  //first check if the alternate is active or not
  if(original_nf == NULL){
    //create buffer for list of pending image (first time)
    nf->image_info.image_pending = (void *) rte_malloc(NULL, sizeof(void *)*MAX_IMAGE, 0);
    nf->image_info.ready_images = (void *) rte_malloc(NULL, sizeof(void* )*MAX_IMAGE, 0);
    nf->image_info.num_of_ready_images = 0;
    nf->image_info.index_of_ready_image = 0;
    
    int i;
    //empty the array
    for(i = 0; i<MAX_IMAGE; i++)
      nf->image_info.image_pending[i] = NULL;
  }
  else
    {
      //attach to the alternate one's buffer
      nf->image_info = original_nf->image_info;
    } 
}
*/

/* helper function to get the image index */
static int get_image_index(image_data *image, image_data **image_list){
  int i;
  for(i = 0; i<MAX_IMAGE; i++){
    if(image_list[i] == image)
      return i;
  }
  return -1;
}

//get a new image from mempool

static image_data *get_image(int pkt_file_index, image_data **pending_img){
  int ret;
  struct image_data *img;
  //check if the image already exist...
  if(pending_img[pkt_file_index] != NULL){
    return pending_img[pkt_file_index];
  }
  //else make a new image
  ret = rte_mempool_get(nf_image_pool,(void **)(&img));
  img->image_id = pkt_file_index; //current logic is to just name the image ID as same as the packet ID. this is not transferrable across the NFs other than the alternate NF
  if(ret != 0){
    RTE_LOG(INFO, APP, "unable to allocate image from pool \n");
    return NULL;
  }

  //now we have to do the accounting... record it..
  pending_img[pkt_file_index] = img;
  return img;
}

//put the images back in the mempool
 static void delete_image(image_data *image, struct onvm_nf_info *nf_info){
   image_data ** image_list = (image_data **)nf_info->image_info->ready_images;
   int retval = get_image_index(image, image_list);//find where the mempool is located
  
  if(retval>=0)
    image_list[retval] = NULL;
  
  rte_mempool_put(nf_image_pool, (void *)image);
}

void copy_data_to_image(void *packet_data,struct onvm_nf_info *nf_info){
  image_data **pending_images = (image_data **)nf_info->image_info->image_pending;
  data_struct *pkt_data = (data_struct *)packet_data;
  image_data *image = get_image(pkt_data->file_id, pending_images);
  memcpy(&(image->image_data_arr[pkt_data->position]), pkt_data->data_array, sizeof(float)*pkt_data->number_of_elements);
  image->num_data_points_stored += pkt_data->number_of_elements;

  //printf("number of data points received %d \n",image->num_data_points_stored);
  //check if the image is ready for evaluation
  if(image->num_data_points_stored >= IMAGE_NUM_ELE){
    //add to the ready image list.
    nf_info->image_info->ready_images[(nf_info->image_info->num_of_ready_images+nf_info->image_info->index_of_ready_image)%MAX_IMAGE] = (void *)image;
    image->status = ready;
    nf_info->image_info->num_of_ready_images++;
    printf("DEBUG.... all packets of image is received \n");
    //remove it from pending image list and put it on 
    pending_images[image->image_id] = NULL;
  }

}

//initializes the timers....
void initialize_ml_timers(struct onvm_nf_info * nf_info){
  rte_timer_subsystem_init();//does this happen already.. check
  rte_timer_init(&image_stats_timer);
  rte_timer_init(&image_inference_timer);
  rte_timer_reset_sync(&image_stats_timer,
		       (NF_IMAGE_STATS_PERIOD_MS * rte_get_timer_hz())/1000,
		       PERIODICAL,
		       rte_lcore_id(), //timer_core
		       (void *)&compute_ml_stats,
		       (void *) nf_info
		       );
  
  rte_timer_reset_sync(&image_inference_timer,
		       (NF_INFERENCE_PERIOD * rte_get_timer_hz())/1000,
		       PERIODICAL,
		       rte_lcore_id(), //timer_core
		       (void *)&evaluate_an_image_from_mempool,
		       (void *) nf_info
		       );
}

void compute_ml_stats(struct onvm_nf_info *nf_info){
  //compute the values in per second basis and feed into histogram.
  #ifdef ONVM_GPU_SAME_SIZE_PKTS
  int number_of_images_pending = nf_info->number_of_pkts_outstanding/NUM_OF_PKTS;
  hist_store_v2(&nf_info->image_queueing_rate, number_of_images_pending);
  nf_info->number_of_pkts_outstanding = 0;
  #endif
  hist_store_v2(&nf_info->image_processing_rate, nf_info->number_of_images_processed);
  nf_info->number_of_images_processed = 0;
}


#endif //ONVM_GPU

static inline void
onvm_nflib_start_nf(struct onvm_nf_info *nf_info){

#ifdef ENABLE_LOCAL_LATENCY_PROFILER
        onvm_util_get_start_time(&ts);
#endif
#ifdef ENABLE_MSG_CONSTRUCT_NF_INFO_NOTIFICATION
        struct onvm_nf_msg *startup_msg;
        /* Put this NF's info struct into queue for manager to process startup shutdown */
        if (rte_mempool_get(nf_msg_pool, (void**)(&startup_msg)) != 0) {
                rte_mempool_put(nf_info_mp, nf_info); // give back mermory
                rte_exit(EXIT_FAILURE, "Cannot create shutdown msg");
        }
        startup_msg->msg_type = MSG_NF_STARTING;
        startup_msg->msg_data = nf_info;
        if (rte_ring_enqueue(mgr_msg_ring, startup_msg) < 0) {
                rte_mempool_put(nf_info_mp, nf_info); // give back mermory
                rte_mempool_put(nf_msg_pool, startup_msg);
                rte_exit(EXIT_FAILURE, "Cannot send nf_info to manager for startup");
        }
#else
        /* Put this NF's info struct onto queue for manager to process startup */
        if (rte_ring_enqueue(mgr_msg_ring, nf_info) < 0) {
                rte_mempool_put(nf_info_mp, nf_info); // give back mermory
                rte_exit(EXIT_FAILURE, "Cannot send nf_info to manager");
        }
#endif

        /* Wait for a client id to be assigned by the manager */
        RTE_LOG(INFO, APP, "Waiting for manager to assign an ID...\n");
        struct timespec req = {0,1000}, res = {0,0};
        for (; nf_info->status == (uint16_t)NF_WAITING_FOR_ID ;) {
                nanosleep(&req,&res);//sleep(1);
        }
#ifdef ENABLE_LOCAL_LATENCY_PROFILER
        int64_t ttl_elapsed = onvm_util_get_elapsed_time(&ts);
        printf("WAIT_TIME(INIT-->START): %li ns\n", ttl_elapsed);
#endif

        /* This NF is trying to declare an ID already in use. */
        if (nf_info->status == NF_ID_CONFLICT) {
                rte_mempool_put(nf_info_mp, nf_info);
                rte_exit(NF_ID_CONFLICT, "Selected ID already in use. Exiting...\n");
        } else if(nf_info->status == NF_NO_IDS) {
                rte_mempool_put(nf_info_mp, nf_info);
                rte_exit(NF_NO_IDS, "There are no ids available for this NF\n");
        } else if(nf_info->status != NF_STARTING) {
                rte_mempool_put(nf_info_mp, nf_info);
                rte_exit(EXIT_FAILURE, "Error occurred during manager initialization\n");
        }
        RTE_LOG(INFO, APP, "Using Instance ID %d\n", nf_info->instance_id);
        RTE_LOG(INFO, APP, "Using Service ID %d\n", nf_info->service_id);

        /* Firt update this client structure pointer */
        this_nf = &nfs[nf_info->instance_id];

        /* Now, map rx and tx rings into client space */
        rx_ring = rte_ring_lookup(get_rx_queue_name(nf_info->instance_id));
        if (rx_ring == NULL)
                rte_exit(EXIT_FAILURE, "Cannot get RX ring - is server process running?\n");

        tx_ring = rte_ring_lookup(get_tx_queue_name(nf_info->instance_id));
        if (tx_ring == NULL)
                rte_exit(EXIT_FAILURE, "Cannot get TX ring - is server process running?\n");

#if defined(ENABLE_SHADOW_RINGS)
        rx_sring = rte_ring_lookup(get_rx_squeue_name(nf_info->instance_id));
        if (rx_sring == NULL)
                rte_exit(EXIT_FAILURE, "Cannot get RX Shadow ring - is server process running?\n");

        tx_sring = rte_ring_lookup(get_tx_squeue_name(nf_info->instance_id));
        if (tx_sring == NULL)
                rte_exit(EXIT_FAILURE, "Cannot get TX Shadow ring - is server process running?\n");
#endif

        nf_msg_ring = rte_ring_lookup(get_msg_queue_name(nf_info->instance_id));
        if (nf_msg_ring == NULL)
                rte_exit(EXIT_FAILURE, "Cannot get nf msg ring");

#ifdef ENABLE_REPLICA_STATE_UPDATE
        pReplicaStateMempool = nfs[get_associated_active_or_standby_nf_id(nf_info->instance_id)].nf_state_mempool;
#endif
}

#ifdef INTERRUPT_SEM
static void set_cpu_sched_policy_and_mode(void) {
        return;

        struct sched_param param;
        pid_t my_pid = getpid();
        sched_getparam(my_pid, &param);
        param.__sched_priority = 20;
        sched_setscheduler(my_pid, SCHED_RR, &param);
}

static void 
init_shared_cpu_info(uint16_t instance_id) {
        const char *sem_name;
        int shmid;
        key_t key;
        char *shm;

        sem_name = get_sem_name(instance_id);
        fprintf(stderr, "sem_name=%s for client %d\n", sem_name, instance_id);

        #ifdef USE_SEMAPHORE
        mutex = sem_open(sem_name, 0, 0666, 0);
        if (mutex == SEM_FAILED) {
                perror("Unable to execute semaphore");
                fprintf(stderr, "unable to execute semphore for client %d\n", instance_id);
                sem_close(mutex);
                exit(1);
        }
        #endif

        /* get flag which is shared by server */
        key = get_rx_shmkey(instance_id);
        if ((shmid = shmget(key, SHMSZ, 0666)) < 0) {
                perror("shmget");
                fprintf(stderr, "unable to Locate the segment for client %d\n", instance_id);
                exit(1);
        }

        if ((shm = shmat(shmid, NULL, 0)) == (char *) -1) {
                fprintf(stderr, "can not attach the shared segment to the client space for client %d\n", instance_id);
                exit(1);
        }

        flag_p = (rte_atomic16_t *)shm;

        set_cpu_sched_policy_and_mode();
}
#endif //INTERRUPT_SEM
