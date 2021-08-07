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
import sys
from subprocess import check_output

MLPERF_LOG_ACCURACY_JSON = 'mlperf_log_accuracy.json'
MLPERF_LOG_DETAIL_TXT    = 'mlperf_log_detail.txt'
MLPERF_LOG_SUMMARY_TXT   = 'mlperf_log_summary.txt'
MLPERF_LOG_TRACE_JSON    = 'mlperf_log_trace.json'
MLPERF_USER_CONF         = 'user.conf'
MLPERF_AUDIT_CONF        = 'audit.config'
ACCURACY_TXT             = 'accuracy.txt'


def ck_postprocess(i):
  print('\n--------------------------------')

  env                   = i['env']
  deps                  = i['deps']
  SIDELOAD_JSON         = env.get('CK_LOADGEN_SIDELOAD_JSON', '')
  include_trace         = env.get('CK_LOADGEN_INCLUDE_TRACE', '') in ('YES', 'Yes', 'yes', 'TRUE', 'True', 'true', 'ON', 'On', 'on', '1')
  LOADGEN_DATASET_SIZE  = env.get('CK_LOADGEN_DATASET_SIZE', '')

  loadgen_dep           = deps['lib-python-loadgen']
  python_dep            = deps.get('python') or loadgen_dep['dict']['deps']['python']
  inference_src_dep     = deps.get('mlperf-inference-src') or loadgen_dep['dict']['deps']['mlperf-inference-src']
  inference_src_env     = inference_src_dep['dict']['env']
  MLPERF_MAIN_CONF      = inference_src_env['CK_ENV_MLPERF_INFERENCE_MLPERF_CONF']
  BERT_CODE_ROOT        = inference_src_env['CK_ENV_MLPERF_INFERENCE']+'/language/bert'
  BERT_MODULES_DIR      = os.path.join( BERT_CODE_ROOT, "DeepLearningExamples", "TensorFlow", "LanguageModeling", "BERT")

  dataset_tokenized_dep = deps['dataset-tokenized']
  dataset_original_dep  = deps.get('dataset-original') or dataset_tokenized_dep['dict']['deps']['dataset-original']
  dataset_vocab_dep     = deps.get('dataset-vocab') or dataset_tokenized_dep['dict']['deps']['dataset-vocab']

  SQUAD_DATASET_ORIGINAL_PATH   = dataset_original_dep['dict']['env']['CK_ENV_DATASET_SQUAD_ORIGINAL']
  SQUAD_DATASET_TOKENIZED_PATH  = dataset_tokenized_dep['dict']['env']['CK_ENV_DATASET_SQUAD_TOKENIZED']
  DATASET_TOKENIZATION_VOCAB    = dataset_vocab_dep['dict']['env']['CK_ENV_DATASET_TOKENIZATION_VOCAB']

  
  save_dict = {}

  # Save logs.
  mlperf_log_dict   = save_dict['mlperf_log'] = {}
  mlperf_conf_dict  = save_dict['mlperf_conf'] = {}

  with open(MLPERF_LOG_ACCURACY_JSON, 'r') as accuracy_file:
    mlperf_log_dict['accuracy'] = json.load(accuracy_file)

  with open(MLPERF_LOG_SUMMARY_TXT, 'r') as summary_file:
    unstripped_summary_lines = summary_file.readlines()
    mlperf_log_dict['summary'] = unstripped_summary_lines

    save_dict['parsed_summary'] = {}
    parsed_summary = save_dict['parsed_summary']
    for line in unstripped_summary_lines:
      pair = line.strip().split(': ', 1)
      if len(pair)==2:
        parsed_summary[ pair[0].strip() ] = pair[1].strip()

  with open(MLPERF_LOG_DETAIL_TXT, 'r') as detail_file:
    mlperf_log_dict['detail'] = detail_file.readlines()

  if include_trace and os.stat(MLPERF_LOG_TRACE_JSON).st_size!=0:
    with open(MLPERF_LOG_TRACE_JSON, 'r') as trace_file:
      mlperf_log_dict['trace'] = json.load(trace_file)
  else:
    mlperf_log_dict['trace'] = {}

  for conf_path in (MLPERF_MAIN_CONF, MLPERF_USER_CONF, MLPERF_AUDIT_CONF):
    if os.path.exists( conf_path ):
      with open(conf_path, 'r') as conf_fd:
        mlperf_conf_dict[ os.path.basename(conf_path) ] = conf_fd.readlines()

  # Check accuracy in accuracy mode.
  # NB: Used to be just (mlperf_log_dict['accuracy'] != []) but this proved
  # to be unreliable with compliance TEST01 which samples accuracy.
  accuracy_mode = (save_dict['parsed_summary'] == {})
  if accuracy_mode:

    ## Combine the PYTHONPATH environments from all deps that might contain it and load it for the accuracy script:
    #
    pp_list = []
    for dep1 in sorted( deps.values(), key=lambda x: int(x['sort']) ):          # work with top level deps
        deps2 = dep1['dict'].get('deps', {})
        for dep2 in sorted( deps2.values(), key=lambda x: int(x['sort']) ):     # also include next level deps (not deeper!)
            dep2_pp = dep2.get('dict',{}).get('env',{}).get('PYTHONPATH')       # some of it may be missing
            if dep2_pp:
                pp_list.append( dep2_pp.split(':')[0] )

        dep1_pp = dep1['dict']['env'].get('PYTHONPATH')
        if dep1_pp:
            pp_list.append( dep1_pp.split(':')[0] )
    pp_list.append( os.environ.get('PYTHONPATH','') )

    pp_list.insert( 0, BERT_MODULES_DIR )

    os.environ['PYTHONPATH'] = ':'.join( pp_list )

    command = [ python_dep['dict']['env']['CK_ENV_COMPILER_PYTHON_FILE'], BERT_CODE_ROOT+'/accuracy-squad.py',
              '--vocab_file', DATASET_TOKENIZATION_VOCAB,
              '--val_data', SQUAD_DATASET_ORIGINAL_PATH,
              '--features_cache_file', SQUAD_DATASET_TOKENIZED_PATH,
              '--log_file', MLPERF_LOG_ACCURACY_JSON,
              '--out_file', 'predictions.json',
    ]
    if LOADGEN_DATASET_SIZE:
        command.extend( [ '--max_examples', str(LOADGEN_DATASET_SIZE) ] )

    output = check_output(command).decode('ascii')

    print(output)

    with open(ACCURACY_TXT, 'w') as accuracy_file:
      accuracy_file.write(output)

    matchObj  = re.match('{"exact_match": ([\d\.]+), "f1": ([\d\.]+)}', output)
    save_dict['exact_match']    = float( matchObj.group(1) )
    save_dict['f1']             = float( matchObj.group(2) )

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

  if SIDELOAD_JSON:
    if os.path.exists(SIDELOAD_JSON):
      with open(SIDELOAD_JSON, 'r') as sideload_fd:
        sideloaded_data = json.load(sideload_fd)
    else:
        sideloaded_data = {}
    save_dict['sideloaded_data'] = sideloaded_data


  with open('tmp-ck-timer.json', 'w') as save_file:
    json.dump(save_dict, save_file, indent=2, sort_keys=True)

  print('--------------------------------\n')
  return {'return': 0}

