#!/bin/bash

curl -X POST http://localhost:8042/tools/execute-script --data-binary @route_dicoms.lua -v
storescp 106 -v -aet HIPPOAI -od /home/workspace/routed_dir --sort-on-study-uid st