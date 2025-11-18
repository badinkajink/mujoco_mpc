#ifndef MJPC_TASKS_HUMANOID_BENCH_AVOID_AVOID_H_
#define MJPC_TASKS_HUMANOID_BENCH_AVOID_AVOID_H_

#include <memory>
#include <random>
#include <string>

#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include "mujoco/mujoco.h"

namespace mjpc {
class avoid : public Task {
 public:
  std::string Name() const override = 0;

  std::string XmlPath() const override = 0;

  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const avoid *task)
        : mjpc::BaseResidualFn(task), task_(const_cast<avoid *>(task)) {}

    void Residual(const mjModel *model, const mjData *data,
                  double *residual) const override;

    private:
      avoid *task_;
      friend class avoid;

  };

  avoid() : residual_(this) {
    target_position_ = {1.2, 0.0, 0.95};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_x(1.1, 1.3);
    std::uniform_real_distribution<> dis_y(-0.3, 0.3);
    target_position_ = {dis_x(gen), dis_y(gen), 0.95};
  }

//   void TransitionLocked(mjModel *model, mjData *data) override;

  void ResetLocked(const mjModel *model) override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const

      override {
    return std::make_unique<ResidualFn>(this);
  }

  ResidualFn *InternalResidual()

      override {
    return &residual_;
  }

private:
  ResidualFn residual_;
};

class Avoid_H12 : public avoid {
 public:
  std::string Name() const override { return "Avoid H12"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/avoid/Avoid_H12.xml");
  }
};


}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_BENCH_AVOID_AVOID_H_