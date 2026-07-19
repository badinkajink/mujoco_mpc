#include "mjpc/deploy/deploy_telemetry.h"

#include <cstdio>

namespace h12deploy {

void AppendPlanJson(const mjpc::Trajectory* traj, int nq, std::int64_t plan_iter,
                    std::string* out) {
  char buf[40];
  auto num = [&](double v) {
    // %.6g keeps rad-scale qpos well inside double round-trip needs for a
    // debug view while holding the payload to ~90 KB.
    int n = std::snprintf(buf, sizeof(buf), "%.6g", v);
    out->append(buf, n > 0 ? n : 0);
  };
  const int H = traj->horizon;
  const int ds = traj->dim_state;

  out->clear();
  out->reserve(static_cast<std::size_t>(H) * nq * 9 + 256);
  out->append("{\"stamp\":");
  num(H > 0 ? traj->times[0] : 0.0);
  out->append(",\"plan_iter\":");
  {
    int n = std::snprintf(buf, sizeof(buf), "%lld",
                          static_cast<long long>(plan_iter));
    out->append(buf, n > 0 ? n : 0);
  }
  out->append(",\"nq\":");
  { int n = std::snprintf(buf, sizeof(buf), "%d", nq); out->append(buf, n > 0 ? n : 0); }
  out->append(",\"horizon\":");
  { int n = std::snprintf(buf, sizeof(buf), "%d", H); out->append(buf, n > 0 ? n : 0); }

  out->append(",\"times\":[");
  for (int t = 0; t < H; t++) {
    if (t) out->push_back(',');
    num(traj->times[t]);
  }
  out->append("],\"qpos\":[");
  for (int t = 0; t < H; t++) {                   // row-major: H rows of nq
    const double* q = traj->states.data() + static_cast<std::size_t>(t) * ds;
    for (int i = 0; i < nq; i++) {
      if (t || i) out->push_back(',');
      num(q[i]);
    }
  }
  out->append("],\"total_return\":");
  num(traj->total_return);
  out->append(",\"failure\":");
  out->append(traj->failure ? "true" : "false");
  out->append("}");
}

}  // namespace h12deploy
