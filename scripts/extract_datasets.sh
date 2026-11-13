#!/usr/bin/env bash
# Verify and extract all LTR datasets into data/raw/<name>/.
# Safe to re-run: a dataset with a .extracted marker is skipped.
# Archives are left untouched.
set -euo pipefail

RAW="$(cd "$(dirname "$0")/../data/raw" && pwd)"
cd "$RAW"

log() { echo "[$(date +%H:%M:%S)] $*"; }

check_md5() {  # file expected
  local got; got=$(md5 -q "$1")
  if [[ "$got" != "$2" ]]; then log "MD5 MISMATCH $1 ($got)"; exit 1; fi
  log "md5 ok: $1"
}

extract() {  # name command...
  local name=$1; shift
  if [[ -f "$name/.extracted" ]]; then log "skip $name (already done)"; return; fi
  log "extracting $name ..."
  mkdir -p "$name"
  "$@"
  touch "$name/.extracted"
  log "done $name"
}

# --- verify ---
check_md5 MSLR-WEB10K.zip 97c5d4e7c171e475c91d7031e4fd8e79
check_md5 MSLR-WEB30K.zip 4beae4bee0cd244fc9b2aff355a61555

# --- extract ---
extract MSLR-WEB10K unzip -q -n MSLR-WEB10K.zip -d MSLR-WEB10K
extract MSLR-WEB30K unzip -q -n MSLR-WEB30K.zip -d MSLR-WEB30K
extract MQ2007      bsdtar -xf MQ2007.rar --strip-components 1 -C MQ2007
extract MQ2008      bsdtar -xf MQ2008.rar --strip-components 1 -C MQ2008
extract istella-s   tar -xzf istella-s-letor.tar.gz -C istella-s
extract istella     tar -xzf istella-letor.tar.gz -C istella

yahoo() {
  tar -xzf yahoo-ltrc-c14-dataset.tgz --strip-components 1 -C yahoo
  tar -xjf yahoo/ltrc_yahoo.tar.bz2 -C yahoo
  cp readme-yahoo-ltrc-c14-dataset.md yahoo/
}
extract yahoo yahoo

istella22() {
  # archive already has a top-level istella22/ folder; inner .gz files stay compressed
  tar -xzf istella22.tar.gz --strip-components 2 -C istella22
  cp test-svm-docIds.gz istella22/
}
extract istella22 istella22

log "all done"
du -sh "$RAW"/*/ 2>/dev/null
