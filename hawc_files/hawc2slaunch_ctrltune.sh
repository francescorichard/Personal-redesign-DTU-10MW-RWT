#!/bin/bash

# Script to submit HAWC2S simulations for all HTC files in a folder
# Usage: ./hawc2slaunch_all.sh <folder_with_htc_files>

if [ $# -eq 0 ]; then
    echo "Error: No folder provided."
    echo "Usage: $0 <folder_with_htc_files>"
    exit 1
fi

FOLDER="$1"

# Check folder exists
if [ ! -d "$FOLDER" ]; then
    echo "Error: Folder does not exist: $FOLDER"
    exit 1
fi

echo "Submitting HAWC2S jobs for all HTC files in: $FOLDER"
echo ""

# Loop through all .htc files in the folder
for HTCFILE in "$FOLDER"/*.htc; do
    
    # Skip if no .htc files exist
    if [ ! -e "$HTCFILE" ]; then
        echo "No .htc files found in folder."
        exit 0
    fi

    echo "Submitting job for: $HTCFILE"

    # Submit the job
    bsub -env "HTCFILE=$HTCFILE" < hawc2slaunch.sh

    # Optional delay
    sleep 1
done

echo ""
echo "All jobs submitted!"
