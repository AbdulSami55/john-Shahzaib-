#!/bin/bash

cd "$(dirname "$0")"
source env/bin/activate
/home/ubuntu/env/bin/uvicorn main:app --host=0.0.0.0 --port=8000