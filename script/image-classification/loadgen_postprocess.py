#
# Copyright (c) 2019 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import os
import json
import re
from subprocess import check_output

MLPERF_LOG_ACCURACY_JSON = 'mlperf_log_accuracy.json'
MLPERF_LOG_DETAIL_TXT    = 'mlperf_log_detail.txt'
MLPERF_LOG_SUMMARY_TXT   = 'mlperf_log_summary.txt'
MLPERF_LOG_TRACE_JSON    = 'mlperf_log_trace.json'

def ck_postprocess(i):
  print('\n--------------------------------')

  env = i['env']

  save_dict = {}

  # Save logs.
  save_dict['mlperf_log'] = {}
  mlperf_log_dict = save_dict['mlperf_log']

  with open(MLPERF_LOG_ACCURACY_JSON, 'r') as accuracy_file:
    mlperf_log_dict['accuracy'] = json.load(accuracy_file)

  with open(MLPERF_LOG_SUMMARY_TXT, 'r') as summary_file:
    mlperf_log_dict['summary'] = summary_file.readlines()

  with open(MLPERF_LOG_DETAIL_TXT, 'r') as detail_file:
    mlperf_log_dict['detail'] = detail_file.readlines()

  if os.stat(MLPERF_LOG_TRACE_JSON).st_size==0:
    mlperf_log_dict['trace'] = {}
  else:
    with open(MLPERF_LOG_TRACE_JSON, 'r') as trace_file:
      mlperf_log_dict['trace'] = json.load(trace_file)


  # Check accuracy in accuracy mode.
  accuracy_mode = False
  if mlperf_log_dict['accuracy'] != []:
    accuracy_mode = True

  if accuracy_mode:
    deps = i['deps']
    accuracy_script = os.path.join( deps['mlperf-inference-src']['dict']['env']['CK_ENV_MLPERF_INFERENCE_V05'],
                                    'classification_and_detection', 'tools', 'accuracy-imagenet.py' )
    imagenet_labels_filepath = deps['imagenet-aux']['dict']['env']['CK_CAFFE_IMAGENET_VAL_TXT']

    command = [ deps['python']['dict']['env']['CK_ENV_COMPILER_PYTHON_FILE'], accuracy_script,
              '--mlperf-accuracy-file', MLPERF_LOG_ACCURACY_JSON,
              '--imagenet-val-file', imagenet_labels_filepath,
              '--dtype', 'float32',
    ]

    output = check_output(command).decode('ascii')

    print(output)
    
    matchObj  = re.match('accuracy=(.+)%, good=(\d+), total=(\d+)', output)
    save_dict['accuracy'] = float( matchObj.group(1) )
    save_dict['good']     = int( matchObj.group(2) )
    save_dict['total']    = int( matchObj.group(3) )
    
  # for scenario in [ 'SingleStream', 'MultiStream', 'Server', 'Offline' ]:
  #   scenario_key = 'TestScenario.%s' % scenario
  #   scenario = save_dict['results'].get(scenario_key, None)
  #   if scenario: # FIXME: Assumes only a single scenario is valid.
  #     save_dict['execution_time_s']  = scenario.get('took', 0.0)
  #     save_dict['execution_time_ms'] = scenario.get('took', 0.0) * 1000
  #     save_dict['percentiles'] = scenario.get('percentiles', {})
  #     save_dict['qps'] = scenario.get('qps', 0)
  #     if accuracy_mode:
  #       ck.out('mAP=%.3f%% (from the results for %s)' % (scenario.get('mAP', 0.0) * 100.0, scenario_key))

  # save_dict['execution_time'] = save_dict['execution_time_s']

  with open('tmp-ck-timer.json', 'w') as save_file:
    json.dump(save_dict, save_file, indent=2, sort_keys=True)

  print('--------------------------------\n')
  return {'return': 0}

