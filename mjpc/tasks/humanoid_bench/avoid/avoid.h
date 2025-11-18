#ifndef MJPC_TASKS_HUMANOID_BENCH_AVOID_AVOID_H_
#define MJPC_TASKS_HUMANOID_BENCH_AVOID_AVOID_H_

#include <memory>
#include <random>
#include <string>

#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include "mujoco/mujoco.h"

namespace mjpc {
class Avoid : public Task {
 public:
  std::string Name() const override = 0;

  std::string XmlPath() const override = 0;

  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Avoid *task)
        : mjpc::BaseResidualFn(task), task_(const_cast<Avoid *>(task)) {}

    void Residual(const mjModel *model, const mjData *data,
                  double *residual) const override;

    private:
      Avoid *task_;
      friend class avoid;

  };

  Avoid() : residual_(this) {
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

class Avoid_H12 : public Avoid {
 public:
  std::string Name() const override { return "Avoid H12"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/avoid/Avoid_H12.xml");
  }
};


}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_BENCH_AVOID_AVOID_H_