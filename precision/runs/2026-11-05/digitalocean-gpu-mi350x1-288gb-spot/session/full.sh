#!/bin/bash
# full.sh <host>: a card's full runs for #34, in order, each through step.sh
H=$1
export ROCM_MAJOR=10
S=~/session/step.sh
bash $S precision-full bash precision/run_by_hand.sh $H
bash $S problems-full bash problems/run_by_hand.sh $H
bash $S problems-highest bash ~/session/highest.sh $H
bash $S repro bash ~/repro/run_repro.sh
