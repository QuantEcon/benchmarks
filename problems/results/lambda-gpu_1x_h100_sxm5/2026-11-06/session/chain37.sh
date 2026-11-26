#!/bin/bash
# chain37.sh <host> [highest label]: #37's full problem-type run and its highest pass, each through step.sh
H=$1
bash ~/session/step.sh problems-full bash problems/run_by_hand.sh $H
bash ~/session/step.sh problems-highest bash ~/session/highest.sh $H ${2:-highest}
