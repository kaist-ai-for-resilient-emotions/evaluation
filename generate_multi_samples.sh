#!/bin/bash

# PYTHONUNBUFFERED=1 python3 gen_virtual.py 7_doyoon --model gpt-normal --client-model fast | tee samples/gpt-normal.fast.7_doyoon.log
PYTHONUNBUFFERED=1 python3 gen_virtual.py 7_doyoon --model gpt-chat --client-model fast | stdbuf -o0 tee samples/gpt-chat.fast.7_doyoon.log