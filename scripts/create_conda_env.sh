#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ENV_ROOT="${1:-$REPO_ROOT/.conda-pi05-openpi-final}"
PKGS_ROOT="${PKGS_ROOT:-/home/S/yangrongzheng/miniconda3/pkgs}"
RLINF_ROOT="${RLINF_ROOT:-/nfs_global/S/yangrongzheng/RLinf-main}"
RLINF_VENV_SITE_PACKAGES="${RLINF_VENV_SITE_PACKAGES:-$RLINF_ROOT/.venv/lib/python3.11/site-packages}"

tmpfile="$(mktemp)"
cleanup() {
  rm -f "$tmpfile"
}
trap cleanup EXIT

cat > "$tmpfile" <<EOF
@EXPLICIT
file://$PKGS_ROOT/_libgcc_mutex-0.1-main.conda
file://$PKGS_ROOT/_openmp_mutex-5.1-1_gnu.conda
file://$PKGS_ROOT/ca-certificates-2025.11.4-h06a4308_0.conda
file://$PKGS_ROOT/ld_impl_linux-64-2.44-h153f514_2.conda
file://$PKGS_ROOT/libgcc-15.2.0-h69a1729_7.conda
file://$PKGS_ROOT/libgcc-ng-15.2.0-h166f726_7.conda
file://$PKGS_ROOT/libgomp-15.2.0-h4751f2c_7.conda
file://$PKGS_ROOT/libstdcxx-15.2.0-h39759b7_7.conda
file://$PKGS_ROOT/libstdcxx-ng-15.2.0-hc03a8fd_7.conda
file://$PKGS_ROOT/bzip2-1.0.8-h5eee18b_6.conda
file://$PKGS_ROOT/expat-2.7.3-h3385a95_0.conda
file://$PKGS_ROOT/libffi-3.4.4-h6a678d5_1.conda
file://$PKGS_ROOT/libnsl-2.0.0-h5eee18b_0.conda
file://$PKGS_ROOT/libuuid-1.41.5-h5eee18b_0.conda
file://$PKGS_ROOT/libxcb-1.17.0-h9b100fa_0.conda
file://$PKGS_ROOT/libzlib-1.3.1-hb25bd0a_0.conda
file://$PKGS_ROOT/ncurses-6.5-h7934f7d_0.conda
file://$PKGS_ROOT/openssl-3.0.18-hd6dcaed_0.conda
file://$PKGS_ROOT/pthread-stubs-0.3-h0ce48e5_1.conda
file://$PKGS_ROOT/readline-8.3-hc2a1206_0.conda
file://$PKGS_ROOT/sqlite-3.51.0-h2a70700_0.conda
file://$PKGS_ROOT/tk-8.6.15-h54e0aa7_0.conda
file://$PKGS_ROOT/tzdata-2025b-h04d1e81_0.conda
file://$PKGS_ROOT/xorg-libx11-1.8.12-h9b100fa_1.conda
file://$PKGS_ROOT/xorg-libxau-1.0.12-h9b100fa_0.conda
file://$PKGS_ROOT/xorg-libxdmcp-1.1.5-h9b100fa_0.conda
file://$PKGS_ROOT/xorg-xorgproto-2024.1-h5eee18b_1.conda
file://$PKGS_ROOT/xz-5.6.4-h5eee18b_1.conda
file://$PKGS_ROOT/zlib-1.3.1-hb25bd0a_0.conda
file://$PKGS_ROOT/python-3.11.14-h6fa692b_0.conda
file://$PKGS_ROOT/pip-25.3-pyhc872135_0.conda
file://$PKGS_ROOT/setuptools-80.9.0-py311h06a4308_0.conda
file://$PKGS_ROOT/wheel-0.45.1-py311h06a4308_0.conda
EOF

rm -rf "$ENV_ROOT"
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  conda create -y -p "$ENV_ROOT" --offline --file "$tmpfile"

mkdir -p "$ENV_ROOT/etc/conda/activate.d" "$ENV_ROOT/etc/conda/deactivate.d"
install -m 0644 \
  "$SCRIPT_DIR/conda_hooks/pi05-openpi-activate.sh" \
  "$ENV_ROOT/etc/conda/activate.d/pi05-openpi.sh"
install -m 0644 \
  "$SCRIPT_DIR/conda_hooks/pi05-openpi-deactivate.sh" \
  "$ENV_ROOT/etc/conda/deactivate.d/pi05-openpi.sh"

cat > "$ENV_ROOT/lib/python3.11/site-packages/rlinf_openpi_bridge.pth" <<EOF
$RLINF_VENV_SITE_PACKAGES
EOF

echo "Created conda env at $ENV_ROOT"
