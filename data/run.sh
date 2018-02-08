#!/bin/sh

for script in prepare*.py; do
	python $script
done
