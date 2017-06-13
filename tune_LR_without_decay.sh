#! /bin/bash

ARGS=("$@")
gpu=${ARGS[0]}

total=${#ARGS[*]}
for (( i=1; i<$total; i++ ))
do
	lr=${ARGS[$i]}
	name=$(printf 'LR_%s_without_decay' "$lr")
	echo "learning rate:" $name
	CUDA_VISIBLE_DEVICES=$gpu th main.lua -name $name -LR $lr -LR_decay_freq 30
done 

