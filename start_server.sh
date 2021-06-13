#!/usr/bin/env bash

nohup python server/run_server.py > $(dirname $(pwd))/chatbot_retrieval/log/server.log 2>&1 &