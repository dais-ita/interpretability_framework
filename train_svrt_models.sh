#!/bin/sh

#for i in $(seq 1 23)

for i in 1 6 17 19 21
do
    python DaisFrameworkTool.py "SVRT Problem $i" vgg16 "" ""
 
done