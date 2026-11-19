#!/bin/bash
# run_all.sh -- the runs behind the figures, then the figures (~8 h).  bash run_all.sh 2>&1 | tee all_log.txt
set -u
cd "$(dirname "$0")"
out=results
run() { python3 run.py run --out $out "$@"; }

CUTS="--cut-at 420 480 540"
run --name turns_x3    --seeds 0-2 --cores 3 $CUTS --cut-n 12                    # 12 of 15 cut at random
run --name wipe_x3     --seeds 0-2 --cores 3 $CUTS --cut-n 12 --cut-what inner   # the inner ring first, 12 in all
run --name inner_x3    --seeds 0-2 --cores 3 $CUTS --cut-n 6 --cut-what inner    # the inner ring only
run --name noturns_x3  --seeds 0-2 --cores 3 $CUTS --cut-n 12 --no-turns         # no turn-taking
run --name free   --seeds 0 --cores 1 --pace 0 &                                   # the rhythm
run --name paced  --seeds 0 --cores 1 &
run --name T10    --seeds 0 --cores 1 --core-T 10 &
wait
run --name T24    --seeds 0 --cores 1 --core-T 24

python3 fig.py turns    $out/turns_x3_s{0,1,2}.json $out/wipe_x3_s{0,1,2}.json --out $out/turns.png
python3 fig.py rhythm   $out/free_s0.json $out/T10_s0.json $out/paced_s0.json $out/T24_s0.json --out $out/rhythm.png
python3 fig.py position $out/turns_x3_s{0,1,2}.json $out/wipe_x3_s{0,1,2}.json $out/noturns_x3_s{0,1,2}.json --out $out/position.png
python3 fig.py levelt   $out/inner_x3_s{0,1,2}.json --out $out/levelt.png
python3 gif.py $out/turns_x3_s0.json --from 400 --to 520 --out $out/cuts.gif --title "12 of 16 tissues cut at 420 and 480"
