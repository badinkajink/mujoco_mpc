#!/usr/bin/env bash
# Build the C++ extensions against the conda `croco` env.
#
#   croco_keepout   box keep-out activation (the S13 speed-up)
#   croco_mfd       matrix-free Delassus benchmark -- reaches
#                   DelassusOperatorRigidBodySystemsTpl, which pinocchio 4.1.0
#                   ships as headers only and does NOT expose to Python
#
# No cmake: these are single translation units against headers that are all in
# one prefix, and a find_package dance would only hide which flags matter.  The
# flags that DO matter:
#
#   -I$PREFIX/include/eigen3      crocoddyl's headers include <Eigen/Dense>
#   -lcrocoddyl                   the extern-template instantiations of
#                                 ActivationModelAbstractTpl<double> live there,
#                                 and linking against a DIFFERENT libcrocoddyl
#                                 than the one crocoddyl's pywrap loaded would
#                                 give two typeinfos and a bases<> that never
#                                 matches at import time
#   -lpinocchio_default -leigenpy the same argument for pinocchio: the operator
#                                 is header-only but its Model/Data/Constraint
#                                 typeinfos have to be the ones the pinocchio
#                                 pywrap already registered, or bp::extract on a
#                                 Python pin.Model finds no converter
#   -lboost_python311             same registry as eigenpy/crocoddyl's bindings
#
# usage: croco_ext/build.sh [--clean] [keepout|mfd]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${CONDA_PREFIX:-$HOME/miniforge3/envs/croco}"
PYVER="$("$PREFIX/bin/python" -c 'import sys; print("%d%d" % sys.version_info[:2])')"
PYDOT="$("$PREFIX/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
EXT="$("$PREFIX/bin/python" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
NPY="$("$PREFIX/bin/python" -c 'import numpy; print(numpy.get_include())')"

if [ "${1:-}" = "--clean" ]; then
  rm -f "$HERE/croco_keepout$EXT" "$HERE/croco_mfd$EXT"
  echo "removed built extensions"; exit 0
fi
WHICH="${1:-all}"

COMMON=(-O3 -DNDEBUG -fPIC -shared -std=c++17
        -I"$PREFIX/include" -I"$PREFIX/include/eigen3"
        -I"$PREFIX/include/python$PYDOT" -I"$NPY"
        -L"$PREFIX/lib" "-lboost_python$PYVER" -Wl,-rpath,"$PREFIX/lib")

if [ "$WHICH" = "all" ] || [ "$WHICH" = "keepout" ]; then
  set -x
  g++ "${COMMON[@]}" "$HERE/keepout.cpp" -o "$HERE/croco_keepout$EXT" -lcrocoddyl
  set +x
  echo "built $HERE/croco_keepout$EXT"
fi

if [ "$WHICH" = "all" ] || [ "$WHICH" = "mfd" ]; then
  # The -D flags are NOT optional and NOT tuning: they are
  # pinocchio::pinocchio_headers' INTERFACE_COMPILE_DEFINITIONS, verbatim from
  # $PREFIX/lib/cmake/pinocchio/pinocchioTargets.cmake.  The constraint
  # collection is a boost::variant over 25 types and the constraint visitors
  # call through 8-argument boost::fusion::invoke; at boost's defaults (20 and
  # 6) the whole header stack fails to instantiate.
  set -x
  g++ "${COMMON[@]}" \
      -DBOOST_MPL_LIMIT_LIST_SIZE=30 -DBOOST_MPL_LIMIT_VECTOR_SIZE=30 \
      -DBOOST_MPL_CFG_NO_PREPROCESSED_HEADERS \
      -DBOOST_FUSION_INVOKE_MAX_ARITY=12 \
      -DPINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION \
      -Wno-deprecated-declarations \
      "$HERE/mfdelassus.cpp" -o "$HERE/croco_mfd$EXT" \
      -lpinocchio_default -leigenpy
  set +x
  echo "built $HERE/croco_mfd$EXT"
fi
