#!/usr/bin/env bash
# Launcher: ./dfl.sh gui | ./dfl.sh swap video.mp4 -m Joker | ./dfl.sh live -m Joker
exec "$(dirname "$(readlink -f "$0")")/venv/bin/dfl" "$@"
