#!/usr/bin/env bash
# Download the trained students into checkpoints/ and verify them.  They are
# attached to the repository's GitHub release, since each is over GitHub's
# 100 MB file limit.
#
#   bash setup/03c_fetch_checkpoints.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

# kept apart so that an anonymising mirror of this repository leaves the URL alone
OWNER=wpfs-anon REPO=WPFS
URL=https://github.com/$OWNER/$REPO/releases/download/checkpoints

# The release holds one flat file per student, so a destination under pi0/ or
# dboft/ is fetched under the prefixed name it carries there.
# sha256                                                            destination                          asset
while read -r sha dst asset; do
  if [ ! -f "$dst" ]; then
    mkdir -p "$(dirname "$dst")"
    echo "downloading $dst"
    curl -fL --retry 3 -o "$dst.part" "$URL/${asset:-$(basename "$dst")}"
    mv "$dst.part" "$dst"
  fi
  echo "$sha  $dst" | sha256sum -c -
done <<'EOF'
7fa94642e578ecf38a5f3fb4af3979394f0c16a21ede67b000d6406e66d68b55 checkpoints/pi05/st_ft_spatial_g0.pt
1bb6b49e8149a38054d551625cf4387841c7ccd36ba298b5f689d391ac3dc5db checkpoints/pi05/st_ft_object_g0.pt
55f0cad1037933c5dc908820133c5e375daa1e562057820e45f841e49424d6b7 checkpoints/pi05/st_ft_goal_g0.pt
93d0ec7b43e99d16cf28adf00d04b64770f5d3026ffb90cc89bc78c621cf8758 checkpoints/pi05/st_long_pe_g0.pt
2dcad9a910c78c9d738bfc407cbdc81b04a4d274115cffc85025458db24e6f88 checkpoints/pi05/st_r8_g0.pt
4ac5c725df936afe744267317c3523c28e5e911a84461a91f5a04e2e01b2a2c4 checkpoints/pi0/student.pt pi0_student.pt
6555180fc3819c853e751b36b7c89fba77287ce4bbaf37f982f920b4c9e16ea8 checkpoints/dboft/student.pt dboft_student.pt
EOF
echo "checkpoints ready"
