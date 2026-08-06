#!/usr/bin/env bash
# Build the C++ box keep-out activation against the conda `croco` env.
#
# No cmake: this is one translation unit against headers that are all in one
# prefix, and a find_package dance would only hide which flags matter.  The
# flags that DO matter:
#
#   -I$PREFIX/include/eigen3      crocoddyl's headers include <Eigen/Dense>
#   -lcrocoddyl                   the extern-template instantiations of
#                                 ActivationModelAbstractTpl<double> live there,
#                                 and linking against a DIFFERENT libcrocoddyl
#                                 than the one crocoddyl's pywrap loaded would
#                                 give two typeinfos and a bases<> that never
#                                 matches at import time
#   -lboost_python311             same registry as eigenpy/crocoddyl's bindings
#
# usage: croco_ext/build.sh [--clean]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${CONDA_PREFIX:-$HOME/miniforge3/envs/croco}"
PYVER="$("$PREFIX/bin/python" -c 'import sys; print("%d%d" % sys.version_info[:2])')"
PYDOT="$("$PREFIX/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
EXT="$("$PREFIX/bin/python" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
OUT="$HERE/croco_keepout$EXT"

if [ "${1:-}" = "--clean" ]; then rm -f "$OUT"; echo "removed $OUT"; exit 0; fi

set -x
g++ -O3 -DNDEBUG -fPIC -shared -std=c++17 \
    -I"$PREFIX/include" -I"$PREFIX/include/eigen3" \
    -I"$PREFIX/include/python$PYDOT" \
    "$HERE/keepout.cpp" -o "$OUT" \
    -L"$PREFIX/lib" -lcrocoddyl "-lboost_python$PYVER" \
    -Wl,-rpath,"$PREFIX/lib"
set +x
echo "built $OUT"
