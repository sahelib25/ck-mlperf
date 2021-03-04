# Description

TODO

# Usage

1. **Store**.
   Run `ck run cmdgen:benchmark.…` with `--scenario=range_singlestream --max_query_count=…` arguments to store performance ranging result to the CK repo.

2. **Extract**.
   Use this script to extract results from the CK repo.
   All arguments are optional.

   ```bash
   $ ck run ck-mlperf:program:generate-target-latency --env.CK_MLPERF_SUBMISSION_REPO=local --env.CK_MLPERF_SUBMISSION_TAGS=foo,bar

   # or, alternatively
   $ ./program/generate-target-latency/run.py --repo-uoa local --tags foo,bar
   ```

   Save the output into the text file.

3. **Use**.
   Run `ck run cmdgen:benchmark.…` with `--target_latency_file=…` instead of `--target_latency=…`.

# Example

```
# Store
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v20.08-neon --model=resnet50 --scenario=range_singlestream --mode=performance --sut=xavier --max_query_count=100
```

```
# Extract
$ ./program/generate-target-latency/run.py | tee target_latency.txt
xavier,armnn-v20.08-neon,resnet50     64 # max_query_count=100
```

```
# Use
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen --library=armnn-v20.08-neon --model=resnet50 --scenario=singlestream --mode=performance --sut=xavier --target_latency_file=target_latency.txt
```
