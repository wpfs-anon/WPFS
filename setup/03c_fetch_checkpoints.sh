#!/usr/bin/env bash
# Download the trained students into checkpoints/ and verify them.  They are
# attached to the repository's GitHub release, since each is over GitHub's
# 100 MB file limit.
#
#   bash setup/03c_fetch_checkpoints.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

URL=https://github.com/wpfs-anon/WPFS/releases/download/checkpoints

# sha256                                                            destination
while read -r sha dst; do
  if [ ! -f "$dst" ]; then
    mkdir -p "$(dirname "$dst")"
    echo "downloading $dst"
    curl -fL --retry 3 -o "$dst.part" "$URL/$(basename "$dst")"
    mv "$dst.part" "$dst"
  fi
  echo "$sha  $dst" | sha256sum -c -
done <<'EOF'
55f0cad1037933c5dc908820133c5e375daa1e562057820e45f841e49424d6b7 checkpoints/pi05/st_ft_goal_g0.pt
1bb6b49e8149a38054d551625cf4387841c7ccd36ba298b5f689d391ac3dc5db checkpoints/pi05/st_ft_object_g0.pt
7fa94642e578ecf38a5f3fb4af3979394f0c16a21ede67b000d6406e66d68b55 checkpoints/pi05/st_ft_spatial_g0.pt
93d0ec7b43e99d16cf28adf00d04b64770f5d3026ffb90cc89bc78c621cf8758 checkpoints/pi05/st_long_pe_g0.pt
2dcad9a910c78c9d738bfc407cbdc81b04a4d274115cffc85025458db24e6f88 checkpoints/pi05/st_r8_g0.pt
933dbbd380ade02308c772af26099f8f557e88071818aa3af66c43dbae0f4935 checkpoints/final30_14-12.pt
724c5a50893ea81625dd064c0c0f5020706344b2d3917859d2781d6cf03cfda8 checkpoints/spnet_g0.pt
EOF
echo "checkpoints ready"
