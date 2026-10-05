#include "mjpc/tasks/humanoid_bench/lean/residual_mlp.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace mjpc {

namespace {
constexpr int kNJ = 27;      // actuated joints, qpos 7.., qvel 6..
constexpr int kNIn = 6 * kNJ + 6;
bool ReadI32(std::ifstream& f, int32_t& v) { return bool(f.read(reinterpret_cast<char*>(&v), 4)); }
bool ReadF32(std::ifstream& f, std::vector<float>& v, size_t n) { v.resize(n); return bool(f.read(reinterpret_cast<char*>(v.data()), 4 * n)); }
}  // namespace

bool ResidualMLP::Load(const std::string& path) {
  enabled_ = false; layers_.clear();
  std::ifstream f(path, std::ios::binary);
  if (!f) { std::fprintf(stderr, "[residual_mlp] cannot open %s\n", path.c_str()); return false; }
  int32_t magic, nl;
  if (!ReadI32(f, magic) || magic != 0x4d4c5031 || !ReadI32(f, nin_) || !ReadI32(f, nout_) || !ReadI32(f, nl) || nl <= 0 || nl > 16) {
    std::fprintf(stderr, "[residual_mlp] bad header in %s\n", path.c_str()); return false;
  }
  for (int l = 0; l < nl; ++l) {
    Layer L; int32_t r, c;
    if (!ReadI32(f, r) || !ReadI32(f, c) || !ReadF32(f, L.W, size_t(r) * c) || !ReadF32(f, L.b, r)) return false;
    L.rows = r; L.cols = c; layers_.push_back(std::move(L));
  }
  if (layers_.front().cols != nin_ || layers_.back().rows != nout_) return false;
  if (!ReadF32(f, mu_, nin_) || !ReadF32(f, sd_, nin_) || !ReadF32(f, ymu_, nout_) || !ReadF32(f, ysd_, nout_) || !ReadF32(f, clip_, nout_)) return false;
  enabled_ = true;
  std::fprintf(stderr, "[residual_mlp] loaded %s: %d -> %d, %d layers\n", path.c_str(), nin_, nout_, nl);
  return true;
}

void ResidualMLP::Forward(const float* x, float* y, std::vector<float>& scratch) const {
  int maxw = nin_;
  for (const auto& L : layers_) maxw = std::max(maxw, L.rows);
  scratch.resize(2 * maxw);
  float* a = scratch.data(); float* b = scratch.data() + maxw;
  for (int i = 0; i < nin_; ++i) a[i] = (x[i] - mu_[i]) / sd_[i];
  for (size_t l = 0; l < layers_.size(); ++l) {
    const Layer& L = layers_[l];
    for (int r = 0; r < L.rows; ++r) {
      const float* w = L.W.data() + size_t(r) * L.cols; float s = L.b[r];
      for (int c = 0; c < L.cols; ++c) s += w[c] * a[c];
      if (l + 1 < layers_.size()) s = s / (1.0f + std::exp(-s));   // SiLU
      b[r] = s;
    }
    std::swap(a, b);
  }
  for (int i = 0; i < nout_; ++i) {
    float v = a[i] * ysd_[i] + ymu_[i];
    y[i] = std::max(-clip_[i], std::min(clip_[i], v));
  }
}

bool H12ResidualFeatures(const mjModel* m, const mjData* d, int imu_site, float* x) {
  if (m->nu != kNJ || m->nq < 7 + kNJ || imu_site < 0) return false;
  const double* q = d->qpos + 7; const double* dq = d->qvel + 6;
  for (int j = 0; j < kNJ; ++j) {
    double kp = m->actuator_gainprm[j * mjNGAIN];            // position actuator: gain kp, bias [0, -kp, -kv]
    double kd = -m->actuator_biasprm[j * mjNBIAS + 2];
    double ctrl = d->ctrl[j];
    double tau = kp * (ctrl - q[j]) - kd * dq[j];
    double lo = m->actuator_forcerange[2 * j], hi = m->actuator_forcerange[2 * j + 1];
    if (m->actuator_forcelimited[j]) tau = std::max(lo, std::min(hi, tau));
    x[j] = float(q[j]); x[kNJ + j] = float(dq[j]); x[2 * kNJ + j] = float(tau); x[3 * kNJ + j] = float(ctrl - q[j]);
    x[4 * kNJ + j] = float(kp / 100.0); x[5 * kNJ + j] = float(kd / 10.0);
  }
  // gravity direction in the IMU frame: R_imu^T * (0,0,-1) = -third row of R  (xmat row-major, world<-site)
  const double* R = d->site_xmat + 9 * imu_site;
  x[6 * kNJ + 0] = float(-R[6]); x[6 * kNJ + 1] = float(-R[7]); x[6 * kNJ + 2] = float(-R[8]);
  x[6 * kNJ + 3] = float(d->qvel[3]); x[6 * kNJ + 4] = float(d->qvel[4]); x[6 * kNJ + 5] = float(d->qvel[5]);
  return true;
}

void H12ApplyResidual(const ResidualMLP& mlp, const mjModel* m, mjData* d, int imu_site, const double group_scale[3]) {
  if (!mlp.enabled() || mlp.nin() != kNIn || mlp.nout() != kNJ) return;
  thread_local std::vector<float> scratch; thread_local float x[kNIn]; thread_local float y[kNJ];
  mj_kinematics(m, d);   // site_xmat is one step stale after mj_step; refresh for the current qpos
  if (!H12ResidualFeatures(m, d, imu_site, x)) return;
  mlp.Forward(x, y, scratch);
  // H12_RESIDUAL_DEBUG=<file>: append the first 200 (features, output, qpos7, qvel6, time) rows for parity checks
  static const char* dbg = std::getenv("H12_RESIDUAL_DEBUG");
  if (dbg) {
    static std::atomic<int> n_dbg{0};
    int k = n_dbg.fetch_add(1);
    if (k < 200) {
      if (FILE* f = std::fopen(dbg, "ab")) {
        double tt = d->time; std::fwrite(&tt, 8, 1, f); std::fwrite(x, 4, kNIn, f); std::fwrite(y, 4, kNJ, f);
        std::fwrite(d->qpos, 8, 7 + kNJ, f); std::fwrite(d->qvel, 8, 6 + kNJ, f); std::fwrite(d->ctrl, 8, kNJ, f); std::fclose(f);
      }
    }
  }
  for (int j = 0; j < kNJ; ++j) {
    double s = j < 12 ? group_scale[0] : (j == 12 ? group_scale[1] : group_scale[2]);
    if (s != 0.0) d->qvel[6 + j] += s * y[j];
  }
}

}  // namespace mjpc

namespace mjpc {

bool ResidualEnsemble::Load(const std::string& paths) {
  members_.clear(); size_t a = 0;
  while (a <= paths.size()) {
    size_t b = paths.find(',', a); std::string one = paths.substr(a, b == std::string::npos ? std::string::npos : b - a);
    if (!one.empty()) { ResidualMLP mlp; if (mlp.Load(one)) members_.push_back(std::move(mlp)); }
    if (b == std::string::npos) break; a = b + 1;
  }
  if (const char* t = std::getenv("H12_RESIDUAL_TAU")) tau_ = std::atof(t);
  if (!members_.empty()) std::fprintf(stderr, "[residual_mlp] ensemble of %zu, tau %.3f rad/s\n", members_.size(), tau_);
  return !members_.empty();
}

void ResidualEnsemble::Apply(const mjModel* m, mjData* d, int imu_site, const double group_scale[3]) const {
  if (members_.empty() || members_[0].nin() != kNIn || members_[0].nout() != kNJ) return;
  thread_local std::vector<float> scratch; thread_local float x[kNIn]; thread_local float y[kNJ];
  thread_local std::vector<float> Y;
  mj_kinematics(m, d);
  if (!H12ResidualFeatures(m, d, imu_site, x)) return;
  const int K = members_.size(); Y.assign(size_t(K) * kNJ, 0.0f);
  for (int k = 0; k < K; ++k) { members_[k].Forward(x, y, scratch); for (int j = 0; j < kNJ; ++j) Y[k * kNJ + j] = y[j]; }
  for (int j = 0; j < kNJ; ++j) {
    double mu = 0, s2 = 0;
    for (int k = 0; k < K; ++k) mu += Y[k * kNJ + j];
    mu /= K;
    for (int k = 0; k < K; ++k) { double e = Y[k * kNJ + j] - mu; s2 += e * e; }
    double sd = K > 1 ? std::sqrt(s2 / K) : 0.0;
    double gate = K > 1 ? std::exp(-(sd / tau_) * (sd / tau_)) : 1.0;
    double s = j < 12 ? group_scale[0] : (j == 12 ? group_scale[1] : group_scale[2]);
    if (s != 0.0) d->qvel[6 + j] += s * gate * mu;
  }
}

}  // namespace mjpc
