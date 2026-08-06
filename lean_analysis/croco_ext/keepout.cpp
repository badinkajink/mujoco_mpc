// Box keep-out activation for crocoddyl, in C++.
//
// WHY.  The keep-out is the OCP's only geometric constraint on the table
// (croco_geom.py explains why a half-space cannot be it), and it is written as
// one activation per sampled point per node.  The S13 plan carries 86 points
// over 290 nodes; as a Python subclass of `crocoddyl.ActivationModelAbstract`
// that is 24 940 interpreter round-trips per calc/calcDiff sweep, and the
// measured cost was the dominant term in both the offline solve (50 s) and the
// receding-horizon MPC (119 ms/step against a 20 ms control period).
//
// The maths is unchanged from croco_geom.ActivationModelBoxKeepOut -- this is a
// transcription, and `croco_ext/test_keepout.py` checks the two agree to 1e-12
// on random residuals, including on the branch boundaries.
//
// ONE THING WORTH KNOWING about the Hessian.  crocoddyl stores `Arr` as an
// Eigen DiagonalMatrix, and the Python binding's setter is
//     data.Arr.diagonal() = Arr.diagonal()
// so the Python model's `data.Arr = np.outer(g, g)` was only ever contributing
// diag(g_i^2) -- the off-diagonal half of the Gauss-Newton term was silently
// discarded by the binding.  This C++ version writes diag(g_i^2) explicitly, so
// it reproduces the Python behaviour exactly rather than "fixing" it into a
// different optimisation problem.
//
// build:  croco_ext/build.sh      (writes croco_keepout*.so next to this file)

#include <boost/python.hpp>

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <sstream>

#include "crocoddyl/core/activation-base.hpp"

namespace bp = boost::python;

namespace {

typedef crocoddyl::ActivationModelAbstractTpl<double> Base;
typedef crocoddyl::ActivationDataAbstractTpl<double> BaseData;

// Signed distance from an already box-centred point to an axis-aligned box.
inline double sdfBox(const Eigen::Vector3d& p, const Eigen::Vector3d& half) {
  const Eigen::Vector3d a = p.cwiseAbs() - half;
  const Eigen::Vector3d amax = a.cwiseMax(0.0);
  return amax.norm() + std::min(a.maxCoeff(), 0.0);
}

// d(sdf)/dp.  Outside the box it is the normalised outward part; inside, the
// axis of least penetration.  Both branches match croco_geom.sdf_box_grad.
inline Eigen::Vector3d sdfBoxGrad(const Eigen::Vector3d& p,
                                  const Eigen::Vector3d& half) {
  const Eigen::Vector3d a = p.cwiseAbs() - half;
  Eigen::Vector3d sg;
  for (int i = 0; i < 3; ++i) sg[i] = (p[i] < 0.0) ? -1.0 : 1.0;
  if (a.maxCoeff() > 0.0) {
    const Eigen::Vector3d amax = a.cwiseMax(0.0);
    const double n = amax.norm();
    return sg.cwiseProduct(amax) / std::max(n, 1e-12);
  }
  Eigen::Vector3d g = Eigen::Vector3d::Zero();
  int i = 0;
  a.maxCoeff(&i);
  g[i] = sg[i];
  return g;
}

class ActivationModelBoxKeepOut : public Base {
 public:
  ActivationModelBoxKeepOut(double hx, double hy, double hz, double r_min)
      : Base(3), half_(hx, hy, hz), r_min_(r_min) {}

  void calc(const std::shared_ptr<BaseData>& data,
            const Eigen::Ref<const Eigen::VectorXd>& r) override {
    const double v = r_min_ - sdfBox(r.head<3>(), half_);
    data->a_value = (v > 0.0) ? 0.5 * v * v : 0.0;
  }

  void calcDiff(const std::shared_ptr<BaseData>& data,
                const Eigen::Ref<const Eigen::VectorXd>& r) override {
    const Eigen::Vector3d p = r.head<3>();
    const double v = r_min_ - sdfBox(p, half_);
    if (v <= 0.0) {
      data->Ar.setZero();
      data->Arr.diagonal().setZero();
      return;
    }
    const Eigen::Vector3d g = sdfBoxGrad(p, half_);
    data->Ar = -v * g;
    data->Arr.diagonal() = g.array().square();
  }

  // Required by CROCODDYL_BASE_CAST.  Only the double instantiation is ever
  // used here; the float scalar exists for the library's codegen paths, which
  // this study does not use, so it clones as double rather than pretending to
  // support a scalar the SDF branches were never checked in.  (The macro also
  // declares cloneAsADDouble, but only in a CppAD-enabled build; this
  // conda-forge crocoddyl is not one, so declaring it here fails to override.)
  std::shared_ptr<crocoddyl::ActivationModelBase> cloneAsDouble() const override {
    return std::make_shared<ActivationModelBoxKeepOut>(half_[0], half_[1],
                                                       half_[2], r_min_);
  }
  std::shared_ptr<crocoddyl::ActivationModelBase> cloneAsFloat() const override {
    return cloneAsDouble();
  }

  void print(std::ostream& os) const override {
    os << "ActivationModelBoxKeepOut {half=[" << half_.transpose()
       << "], r_min=" << r_min_ << "}";
  }

  const Eigen::Vector3d& get_half() const { return half_; }
  double get_r_min() const { return r_min_; }

 private:
  Eigen::Vector3d half_;
  double r_min_;
};

// Standalone SDF exports, so the Python side can check the two implementations
// agree without going through an activation data.
double py_sdf(double x, double y, double z, double hx, double hy, double hz) {
  return sdfBox(Eigen::Vector3d(x, y, z), Eigen::Vector3d(hx, hy, hz));
}

bp::tuple py_sdf_grad(double x, double y, double z, double hx, double hy,
                      double hz) {
  const Eigen::Vector3d g =
      sdfBoxGrad(Eigen::Vector3d(x, y, z), Eigen::Vector3d(hx, hy, hz));
  return bp::make_tuple(g[0], g[1], g[2]);
}

// calc/calcDiff take `Eigen::Ref<const VectorXd>` because that is the signature
// the crocoddyl base declares.  Binding those directly does NOT work: eigenpy's
// converter for a const Ref hands boost::python a temporary whose storage is
// gone by the time the call is made, and the observable symptom is a stack-smash
// abort with no traceback, not a type error.  These thunks take a VectorXd by
// const reference -- a plain, well-supported eigenpy conversion -- and forward.
void py_calc(ActivationModelBoxKeepOut& a, const std::shared_ptr<BaseData>& d,
             const Eigen::VectorXd& r) {
  a.calc(d, r);
}

void py_calc_diff(ActivationModelBoxKeepOut& a,
                  const std::shared_ptr<BaseData>& d, const Eigen::VectorXd& r) {
  a.calcDiff(d, r);
}

std::string py_repr(const ActivationModelBoxKeepOut& a) {
  std::ostringstream os;
  a.print(os);
  return os.str();
}

}  // namespace

BOOST_PYTHON_MODULE(croco_keepout) {
  // The base class is registered by crocoddyl's own pywrap module, so importing
  // it here is what makes `bp::bases<Base>` resolvable.  Without it boost
  // python raises at REGISTRATION time with an unhelpful message about an
  // unregistered base, which is a confusing way to learn about an import order.
  bp::import("crocoddyl");

  // No Eigen in the constructor signature on purpose: the half extents come in
  // as three doubles, so this module needs no eigenpy numpy converters at all.
  // The hot path (calc/calcDiff) is called from C++ and never crosses numpy.
  bp::class_<ActivationModelBoxKeepOut, bp::bases<Base>,
             std::shared_ptr<ActivationModelBoxKeepOut>, boost::noncopyable>(
      "ActivationModelBoxKeepOut",
      "0.5 * max(0, r_min - sdf_box(r))^2 on a 3-vector residual.\n\n"
      "Pair with ResidualModelFrameTranslation(state, fid, box_centre, nu) so\n"
      "the residual handed in is the point relative to the box centre.",
      bp::init<double, double, double, double>(
          bp::args("self", "hx", "hy", "hz", "r_min")))
      // calc / calcDiff / createData are re-exported HERE rather than inherited.
      // crocoddyl registers them on its Python *wrapper* subclass, not on the
      // abstract base, so a plain C++ derivative inherits the base class in the
      // bp registry but none of its methods -- `createData()` on this object
      // raises an ArgumentError naming ActivationModelAbstractTpl_wrap.  It only
      // matters for direct Python calls (the solver drives calc/calcDiff in C++
      // through the virtual), but "only matters for the test" is exactly the
      // thing a test needs.
      .def("calc", &py_calc, bp::args("self", "data", "r"))
      .def("calcDiff", &py_calc_diff, bp::args("self", "data", "r"))
      .def("createData", &ActivationModelBoxKeepOut::createData,
           bp::args("self"))
      .add_property("r_min", &ActivationModelBoxKeepOut::get_r_min)
      .def("__repr__", &py_repr);

  bp::def("sdf_box", &py_sdf, bp::args("x", "y", "z", "hx", "hy", "hz"));
  bp::def("sdf_box_grad", &py_sdf_grad,
          bp::args("x", "y", "z", "hx", "hy", "hz"));
}
