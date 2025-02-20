#!/bin/bash


if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi


TOKEN_ORDER=(4 5 6 1 2 3)

loop_tokens() {
    local order_length=${#TOKEN_ORDER[@]}
    
    while true; do
        for ((i=0; i<$order_length; i++)); do
            index=$(( ( i) % order_length ))
            token_index=${TOKEN_ORDER[$index]}
            export REAL_TOKEN=$(eval echo \$GITHUB_PAT_v$token_index)
            # echo "Using token $token_index: $REAL_TOKEN"
            python files.py
            echo "sleeping 300..."
            sleep 300
        done
        echo "sleeping 1h..."
        sleep 3600
    done
}


caffeinate -is &
loop_tokens
killall caffeinate