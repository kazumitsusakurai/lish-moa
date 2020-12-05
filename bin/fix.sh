#!/usr/bin/env bash

path=.

if [ "$#" -ge 1 ]; then
  path=$1
fi

autoflake --recursive --in-place --remove-all-unused-imports --ignore-init-module-imports --remove-unused-variables $path
autopep8 --in-place --aggressive --aggressive --recursive --verbose $path
