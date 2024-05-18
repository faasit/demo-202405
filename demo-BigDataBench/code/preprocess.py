import sys
import yaml
import os
# For terasort, it needs replicas, and specifying 
# the DAG and profiles with N*M edges is complicated.
# This script is to extend a yaml(need not be changed except fot mapper number and reducer number) into
# a yaml usable by controller.py

# CAUTION: Only support those like terasort.
# Linear dependency, every replica of previous stage will output to all replicas in the next stage.
# The next stage should wait until all the copies from previous stage are received.

STORAGE_DIR = 'local_storage'

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <input yaml name>",file=sys.stderr)
        exit(1)
    input_yaml_name = sys.argv[1]
    with open(input_yaml_name,'r') as f:
        ydict = yaml.safe_load(f)
    # Check for linear dependency.
    dag = ydict['DAG']
    # Only one stage should have no dependency.
    # Other stages should have one dependency.
    # Only one stage should have no stage depending on it.
    # Other stages should have one stage depending on it.
    head_name = None
    tail_name = None
    stages = {}
    for stage, dep in dag.items():
        if len(dep) == 0:
            if head_name is not None:
                print("More than one head stage.",file=sys.stderr)
                exit(1)
            head_name = stage
        elif len(dep) == 1:
            if dep[0] in stages:
                print("More than one stage depending on one stage.",file=sys.stderr)
                exit(1)
            else:
                stages[dep[0]] = {"next":stage}
        else:
            print("More than one dependency.",file=sys.stderr)
            exit(1)

    for stage in dag.keys():
        if stage not in stages:
            if tail_name != None:
                print("More than one tail stage.",file=sys.stderr)
                exit(1)
            stages[stage] = {"next":None}
            tail_name = stage

    for stage in dag.keys():
        stages[stage]['replicas'] = ydict['stage_profiles'][stage]['replicas']
    
    global_port_alloc = 51000 # Start from 51000 to avoid conflicts.
    global_cache_port_alloc = 59999 # Start from 59999 to avoid conflicts.
    initial_partition_num = int(stages[head_name]['replicas'])
    stage_profiles = {}
    # Not considering the redis overhead for now.
    for stage in stages:
        replica_num = int(stages[stage]['replicas'])
        for i in range(replica_num):
            new_stage_name = stage + '-' + str(i)
            this_stage_info = {}
            this_stage_info['request'] = {}
            for key, value in ydict['stage_profiles'][stage]['request'].items():
                this_stage_info['request'][key] = value
            this_stage_info['input_time'] = ydict['stage_profiles'][stage]['input_time']
            this_stage_info['compute_time'] = ydict['stage_profiles'][stage]['compute_time']
            this_stage_info['output_time'] = ydict['stage_profiles'][stage]['output_time']
            this_stage_info['command'] = ydict['stage_profiles'][stage]['command']
            this_stage_info['args'] = ydict['stage_profiles'][stage]['args']
            this_stage_info['image'] = ydict['stage_profiles'][stage]['image']
            this_stage_info['port'] = global_port_alloc
            this_stage_info['cache_port'] = global_cache_port_alloc
            global_cache_port_alloc -= 1
            global_port_alloc += 1
            stage_profiles[new_stage_name] = this_stage_info

    default_params = {}
    assert(len(stages) > 0)
    input_source_data_name = ydict['input_source_data']
    # BigDataBench generates words which will not go across a line.
    # So we can count the newlines.W
    os.system(f'mkdir -p ./{STORAGE_DIR}')
    with open(input_source_data_name,'r') as f:
        if len(stages) == 2:
            num_mappers = stages[head_name]['replicas']
            num_reducers = stages[tail_name]['replicas']
            assert(num_mappers > 0)
            assert(num_reducers > 0)
            lines = f.readlines()
            total_lines = len(lines)
            chunk_line_size = total_lines // num_mappers
            no = 0
            for i in range(0,total_lines,chunk_line_size):
                oriend = i + chunk_line_size
                end_idx = oriend if oriend <= total_lines else total_lines
                templines = lines[i:end_idx]
                tempstr = "\n".join(templines)
                with open(f"./{STORAGE_DIR}/{head_name}-{no}-input","w") as wrto:
                    wrto.write(tempstr)
                no += 1
            # Then generate the final yaml.
            del ydict['input_source_data']
            del ydict['DAG']
            ydict['stage_profiles'] = stage_profiles
            # DAG and default params.
            # Complete bipartite graph
            # Every second stage should depend on N first stages.
            real_dag = {}
            first_stages = set()
            for stage in stage_profiles.keys():
                full_len = len(stage)
                pt = full_len - 1
                while stage[pt] != '-':
                    pt -= 1
                pure_stage = stage[0:pt]
                if pure_stage == head_name:
                    first_stages.add(stage)
            for stage in stage_profiles.keys():
                if stage in first_stages:
                    real_dag[stage] = []
                else:
                    real_dag[stage] = []
                    for temps in first_stages:
                        real_dag[stage].append(temps)
            ydict['DAG'] = real_dag
            dfp = {}
            for stage in stage_profiles.keys():
                dfp[stage] = {"num_mappers":num_mappers,"num_reducers":num_reducers}
            ydict['default_params'] = dfp
            with open('wordcount.yaml','w') as wrto:
                yaml.dump(ydict,stream=wrto)
        else:
            print(f"len(stages) is {len(stages)} != 2, this pattern has not been implemented.")
            exit(1)
        
        