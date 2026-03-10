#!/bin/bash

declare -A useCases

# Populate the array with use case descriptions and their specified input/output lengths
#useCases["Translation"]="200/200"
#useCases["Text classification"]="200/5"
useCases["Text summary"]="5000/400"

# Function to execute AIPerf with the input/output lengths as arguments
runBenchmark() {
   local description="$1"
   local lengths="${useCases[$description]}"
   IFS='/' read -r inputLength outputLength <<< "$lengths"

   echo "Running AIPerf for $description with input length $inputLength and output length $outputLength"
   #Runs
   for concurrency in 1 2 5 10 50 100 250; do

       local INPUT_SEQUENCE_LENGTH=$inputLength
       local INPUT_SEQUENCE_STD=0
       local OUTPUT_SEQUENCE_LENGTH=$outputLength
       local CONCURRENCY=$concurrency
       local REQUEST_COUNT=$(($CONCURRENCY * 3))
       local MODEL="gemma-3-4b-it-fp8"

       aiperf profile \
           -m $MODEL \
           --endpoint-type chat \
           --streaming \
           -u localhost:8000 \
           --synthetic-input-tokens-mean $INPUT_SEQUENCE_LENGTH \
           --synthetic-input-tokens-stddev $INPUT_SEQUENCE_STD \
           --concurrency $CONCURRENCY \
           --request-count $REQUEST_COUNT \
           --output-tokens-mean $OUTPUT_SEQUENCE_LENGTH \
           --extra-inputs min_tokens:$OUTPUT_SEQUENCE_LENGTH \
           --extra-inputs ignore_eos:true \
           --tokenizer /workspace/gemma-3-4b-it/ \
           --artifact-dir artifact/ISL${INPUT_SEQUENCE_LENGTH}_OSL${OUTPUT_SEQUENCE_LENGTH}/CON${CONCURRENCY}

   done
}

# Iterate over all defined use cases and run the benchmark script for each
for description in "${!useCases[@]}"; do
   runBenchmark "$description"
done
