// Copyright 2024 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MJPC_PLANNERS_IDTO_SETTINGS_H_
#define MJPC_PLANNERS_IDTO_SETTINGS_H_

namespace mjpc {

// Settings for IDTO optimizer
struct IDTOSettings {
  // ----- Solver parameters ----- //

  // Maximum number of iterations per solve
  int max_iterations = 100;

  // Gradient norm convergence tolerance
  double gradient_tolerance = 1.0e-6;

  // Cost change convergence tolerance
  double cost_tolerance = 1.0e-6;

  // ----- Trust region parameters ----- //

  // Initial trust region radius
  double trust_region_radius_initial = 1.0;

  // Minimum trust region radius
  double trust_region_radius_min = 1.0e-6;

  // Maximum trust region radius
  double trust_region_radius_max = 1.0e3;

  // Trust region expansion threshold (increase radius if ratio > threshold)
  double trust_region_expand_threshold = 0.75;

  // Trust region shrink threshold (decrease radius if ratio < threshold)
  double trust_region_shrink_threshold = 0.25;

  // Trust region expansion factor
  double trust_region_expand_factor = 2.0;

  // Trust region shrink factor
  double trust_region_shrink_factor = 0.5;

  // ----- Regularization ----- //

  // Initial Hessian regularization
  double regularization_initial = 1.0e-6;

  // Minimum regularization
  double regularization_min = 1.0e-12;

  // Maximum regularization
  double regularization_max = 1.0e12;

  // Regularization increase factor
  double regularization_increase_factor = 10.0;

  // Regularization decrease factor
  double regularization_decrease_factor = 10.0;

  // ----- Cost weights ----- //

  // State error weight (position)
  double weight_position = 1.0;

  // State error weight (velocity)
  double weight_velocity = 1.0;

  // Control effort weight
  double weight_control = 1.0e-3;

  // Terminal state error weight (position)
  double weight_position_terminal = 1.0;

  // Terminal state error weight (velocity)
  double weight_velocity_terminal = 1.0;

  // ----- Performance options ----- //

  // Number of threads for parallel computation
  int num_threads = 1;

  // Use Gauss-Newton approximation (vs exact Hessian)
  bool use_gauss_newton = true;

  // Real-time iteration mode (limit iterations for MPC)
  bool real_time_mode = false;

  // Maximum iterations in real-time mode
  int real_time_max_iterations = 2;

  // ----- Debugging ----- //

  // Print iteration info
  bool verbose = false;

  // Check gradient and Hessian via finite differences
  bool check_derivatives = false;

  // Finite difference epsilon for derivative checks
  double finite_difference_epsilon = 1.0e-6;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_IDTO_SETTINGS_H_
