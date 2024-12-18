/*
 * clipper_batchsize_extension.c
 *
 *  Created on: Dec 20, 2019
 *      Author: root
 */
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <cinttypes>
#include <dlib/matrix.h>
#include <dlib/svm.h>
#include <unordered_map>


extern "C"{
#include "clipper_batchsize_extension.h"
}
//The minimum number of latency entries associated with a batch
  // size that must exist in order to continue exploration
  // or incorporate variance data
  static constexpr uint32_t MINIMUM_BATCH_SAMPLE_SIZE = 5;
  static constexpr uint32_t LATENCY_Z_SCORE = 3;
  static constexpr double REGRESSION_DATA_SCALE_FACTOR = .001;
  static constexpr size_t ADDITIVE_EXPANSION_THRESHOLD = 10;

  constexpr int DEFAULT_BATCH_SIZE = -1;


  // Exploration and estimation parameters
    // for adaptive batching
    double explore_dist_mu_ = 0.1;
    double explore_dist_std_ = 0.05;
    double budget_decay_ = 0.9;
    //double budget_decay_ = 1.0;
    std::normal_distribution<double> exploration_distribution_(std::normal_distribution<double>(explore_dist_mu_,explore_dist_std_));
    std::default_random_engine exploration_engine_(std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count()));
    //exploration_distribution_
    //exploration_engine_;


enum class BatchSizeDeterminationMethod {
      Default = 0,
      Exploration = 1,
      Estimation = 2
 };

namespace IterativeUpdater {

  /**
   * Given information about a data set's previous mean, std, and cardinality,
   * as well as a new value, calculates the mean and std of the data set augmented
   * by the new value
   *
   * @return A pair consisting of (augmented_mean, augmented_std)
   */
  inline std::pair<double, double> calculate_new_mean_std(double prev_num_samples,
							  double prev_mean,
							  double prev_std,
							  double new_value) {
    double new_num_samples = prev_num_samples + 1;
    //printf("New num samples %f\n",new_num_samples);
      double new_mean =
	((prev_num_samples * prev_mean) + new_value) / new_num_samples;

      double old_s = std::pow(prev_std, 2) * prev_num_samples;
      double new_s = old_s + ((new_num_samples / std::max(1.0, prev_num_samples)) *
			      std::pow((new_mean - new_value), 2));
      double new_std = std::sqrt(new_s / std::max(1.0, new_num_samples));

      return std::make_pair(new_mean, new_std);
  }
}  // namespace IterativeUpdater





// pair of batch size, method by which the batch size was determined
using BatchSizeInfo = std::pair<size_t, BatchSizeDeterminationMethod>;

/* dependencies of the clipper program */
using EstimatorLatency = dlib::matrix<double, 1, 1>;
using EstimatorBatchSize = double;
using EstimatorKernel = dlib::linear_kernel<EstimatorLatency>;
using Estimator = dlib::decision_function<EstimatorKernel>;
// Tuple of num latencies, mean latency, latency std
using LatencyInfo = std::tuple<double, double, double>;


std::unordered_map<EstimatorBatchSize, LatencyInfo> processing_datapoints_;

long long max_latency_ = 0;
  Estimator estimator_;
  dlib::rr_trainer<EstimatorKernel> estimator_trainer_;

size_t max_batch_size_ = 1;
size_t batch_size_ = DEFAULT_BATCH_SIZE;

LatencyInfo update_mean_std(LatencyInfo &info, double new_latency){
	double info_size = std::get<0>(info);
	  double mu = std::get<1>(info);
	  double std = std::get<2>(info);

	  double new_mu;
	  double new_std;
	  std::tie(new_mu, new_std) =
	      IterativeUpdater::calculate_new_mean_std(info_size, mu, std, new_latency);

	  //printf("New latency %f and new mean %f \n",new_latency, new_mu);

	  return std::make_tuple(info_size + 1, new_mu, new_std);

}
void fit_estimator() {
  //std::unique_lock<std::mutex> estimator_lock(estimator_mtx_);
  //boost::shared_lock<boost::shared_mutex> datapoints_lock(datapoints_mtx_);

  size_t num_datapoints = processing_datapoints_.size();

  std::vector<EstimatorLatency> x_vals;
  x_vals.reserve(num_datapoints);
  std::vector<EstimatorBatchSize> y_vals;
  y_vals.reserve(num_datapoints);

  // Calculate the pooled processing latency
  // variance across all batch sizes. This will
  // be used to obtain an estimate for the p99
  // processing latency at each batch size
  //
  // Pooled variance (https://en.wikipedia.org/wiki/Pooled_variance)
  // is used under the empirically tested assumption that latency variance
  // does not change significantly with batch size

  double pooled_std_num = 0;
  double pooled_std_denom = -1 * static_cast<double>(num_datapoints);

  for (auto &entry : processing_datapoints_) {
    LatencyInfo &latency_info = entry.second;
    double info_size = std::get<0>(latency_info);
    if (info_size >= MINIMUM_BATCH_SAMPLE_SIZE) {
      double lats_std = std::get<2>(latency_info);
      double lats_var = std::pow(lats_std, 2);
      pooled_std_num += (info_size - 1) * lats_var;
      pooled_std_denom += info_size;
    }
  }

  double pooled_std =
      std::sqrt(pooled_std_num / std::max(1.0, pooled_std_denom));

  // Using the pooled variance, obtain the
  // estimated p99 latency for each batch size

  EstimatorLatency fitting_lat;
  for (auto &entry : processing_datapoints_) {
    EstimatorBatchSize batch_size = entry.first;
    LatencyInfo &latency_info = entry.second;
    double lats_mean = std::get<1>(latency_info);
    double upper_bound_lat = lats_mean + (LATENCY_Z_SCORE * pooled_std);
    // Scale the upper bound latency in order to moderate the regularization
    // term associated with ridge regression
    fitting_lat(0) = upper_bound_lat * REGRESSION_DATA_SCALE_FACTOR;
    x_vals.push_back(fitting_lat);
    y_vals.push_back(std::move(batch_size));
  }

  //datapoints_lock.unlock();

  estimator_ = estimator_trainer_.train(x_vals, y_vals);
}


extern "C"
void clipper_add_processing_datapoint(
    size_t batch_size, long long processing_latency_micros) {
  if (batch_size <= 0 || processing_latency_micros <= 0) {
    throw std::invalid_argument(
        "Invalid processing datapoint: Batch size and latency must be "
        "positive.");
  }

  //latency_hist_.insert(processing_latency_micros);

//boost::unique_lock<boost::shared_mutex> lock(datapoints_mtx_);

  auto bs_search = processing_datapoints_.find(batch_size);
  if (bs_search == processing_datapoints_.end()) {
    //printf("Batch size %zd is not found \n", batch_size);
    double info_size = 1;
    double std = 0;
    LatencyInfo latency_info = std::make_tuple(
        std::move(info_size), static_cast<double>(processing_latency_micros),
        std::move(std));
    processing_datapoints_.emplace(static_cast<EstimatorBatchSize>(batch_size),
                                   std::move(latency_info));
  } else {
    LatencyInfo &old_info = bs_search->second;
    LatencyInfo new_info = update_mean_std(
        old_info, static_cast<double>(processing_latency_micros));
    
    processing_datapoints_[batch_size] = std::move(new_info);
  }

  //printf("processing latency micros %lld and max latency %lld \n", processing_latency_micros, max_latency_);
  max_latency_ = std::max(processing_latency_micros, max_latency_);
  

  //EstimatorFittingThreadPool::submit_job(model_, replica_id_, [
  //  this, container_activity_mtx = activity_mtx_, container_active = active_
  //]() {
  // std::lock_guard<std::mutex> activity_lock(*container_activity_mtx);
  // if (!(*container_active)) {
  //    return;
  //  }
    try {
      fit_estimator();
    } catch (std::exception const &ex) {
      //log_error_formatted(LOGGING_TAG_CONTAINERS,
      //                     "Error fitting batch size estimator: {}", ex.what());
      printf("Error trying to fit estimator \n");
    }
    //});
}



//this function is called when the latency is much lower than deadline...
// it just increases the batch size by 1/
size_t explore(/* deadline, current batch size*/ /* histogram */){
	/* if not enough sample size in histogram, return the original batch size */
	auto mb_search =
	      processing_datapoints_.find(static_cast<double>(max_batch_size_));
	  if (mb_search != processing_datapoints_.end() &&
	      std::get<0>(mb_search->second) < MINIMUM_BATCH_SAMPLE_SIZE) {
	    // We don't have a large enough latency sample
	    // corresponding to the maximum batch size, so
	    // we won't update the maximum size
	    return max_batch_size_;
	  } else if (max_batch_size_ < ADDITIVE_EXPANSION_THRESHOLD) {
	    return max_batch_size_ + 1;
	  } else {
	    double expansion_factor = exploration_distribution_(exploration_engine_);
	    expansion_factor = std::max(0.0, expansion_factor);
	    return static_cast<size_t>((1 + expansion_factor) * max_batch_size_);
	  }
}

size_t estimate( uint64_t budget ){
	 //std::lock_guard<std::mutex> lock(estimator_mtx_);
	  EstimatorLatency estimator_budget;
	  estimator_budget(0) =
	      static_cast<double>(budget) * REGRESSION_DATA_SCALE_FACTOR;
	  double estimate_ = estimator_(estimator_budget);
	  return static_cast<size_t>(std::max(1.0, std::floor(estimate_)));
}


BatchSizeInfo get_batch_size(double budget /* difference between slo and latency */) {
  BatchSizeDeterminationMethod method;
  if (batch_size_ != DEFAULT_BATCH_SIZE) {
    method = BatchSizeDeterminationMethod::Default;
    return std::make_pair(batch_size_, method);
  }

  //  double budget =
  //    static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
  //                            deadline - std::chrono::system_clock::now())
  //                            .count());
  // Decay the provided latency budget by a pre-specified factor
  // in order to provide enough slack for delivering the response
  // to the user and/or coping with an anomolously high
  // processing latency
  budget = budget * budget_decay_;

  size_t curr_batch_size;
  if (budget > static_cast<double>(max_latency_)) {
    //printf("budget is more than max latency\n");
    curr_batch_size = explore();
    method = BatchSizeDeterminationMethod::Exploration;
  } else {
	  //printf("Estimate called \n");
    curr_batch_size = estimate(budget);
    method = BatchSizeDeterminationMethod::Estimation;
  }
  max_batch_size_ = std::max(curr_batch_size, max_batch_size_);
  return std::make_pair(curr_batch_size, method);
}

extern "C"
int clipper_check_batch_size(long deadline_us)
{
  //printf("Deadline %ld micro-sec \n", deadline_us);
	///just main function to see what libraries I need to compile this with
	//printf("Hello world \n");
	//int i =0;
	//for(i = 1; i < 200; i++){
	  //printf("Max latency %lld \n",max_latency_);
	 // if(!(i%20))
	 //   i = i+1;
	  //add_processing_datapoint(i%20,(i%20)*3*1000);
	  //BatchSizeInfo bs = get_batch_size(70*1000-(i%20)*3*1000);

	  //printf("batch size predicted %zd\n",bs.first);
  //	printf("Max latency %lld and curr_batch_size %d\n",max_latency_, batch_size_);
	BatchSizeInfo bs = get_batch_size(deadline_us);
	return (int)(bs.first);
	
}
