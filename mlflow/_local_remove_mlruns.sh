#!/bin/bash

echo "Permanently remove mlruns folder..."

# Blow away mlruns artifact direcotory

cd ../mlflow
/bin/rm -f -r mlruns && /bin/rm -f -r mlruns.db
