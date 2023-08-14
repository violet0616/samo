"""
This is the main command-line interface for performing optimisation on a CNN
model. Example usage is shown below:

```shell
usage: python -m samo [-h] --model PATH -b {fpgaconvnet,finn,hls4ml} -p PATH
        -o PATH [--optimiser {brute,annealing,init,rule}]
        [--objective {throughput,latency}] [--enable_reconf {true,false}]
        [--seed N] [--batch-size N]

optional arguments:
  -h, --help            show this help message and exit
  -m PATH, --model PATH
                        path to the CNN model that you wish to
                        optimise (.keras, .onnx)
  -b {fpgaconvnet,finn,hls4ml}, --backend {fpgaconvnet,finn,hls4ml}
                        target backend for accelerating the model
  -p PATH, --platform PATH
                        hardware platform details (.json)
  -o PATH, --output-path PATH
                        output path for the optimised model (.json, .onnx)
  --optimiser {brute,annealing,init,rule}
                        optimiser to use
  --objective {throughput,latency}
                        Optimiser objective
  --enable_reconf {true,false}
                        multiple partitions
  --seed N              Seed for the optimiser run
  --batch-size N        batch size
```
"""

import logging
import argparse
import importlib
import json
import copy
import time
import random
import sys
import itertools
import numpy as np
import statistics

import sys
sys.path.append('C:\\IC\\2023_project\\2023_project\\FpgaConvnet_forked\\fpgaconvnet-tutorial\\samo\\samo')  # Add the path to the folder containing the cli script

from samo.optimiser.annealing import SimulatedAnnealing
from optimiser.rule import RuleBased
from samo.optimiser.brute import BruteForce

MY_HUFFMAN_CONSTANT = 3


def main_for_multi_plus_Huffman(args):
    # logging configuration
    logging.basicConfig(filename='samo.log', filemode='w', level=logging.INFO)

    # parse arguments
    parser = argparse.ArgumentParser(description="SAME CNN optimiser")
    parser.add_argument("-m", "--model", metavar="PATH", required=True,
            help="path to the CNN model that you wish to optimise (.keras, .onnx)")
    parser.add_argument("-m1", "--model1", metavar="PATH1", required=True,
            help="path to the CNN model that you wish to optimise (.keras, .onnx)")
    parser.add_argument("-m2", "--model2", metavar="PATH2", required=True,
            help="path to the CNN model that you wish to optimise (.keras, .onnx)")
    parser.add_argument("-m3", "--model3", metavar="PATH3", required=True,
            help="path to the CNN model that you wish to optimise (.keras, .onnx)")
    parser.add_argument("-b", "--backend", choices=["fpgaconvnet", "finn", "hls4ml"], required=True,
            help="target backend for accelerating the model")
    parser.add_argument("-p", "--platform", metavar="PATH", required=True,
            help="hardware platform details (.json)")
    parser.add_argument("-o", "--output-path", metavar="PATH", required=True,
            help="output path for the optimised model (.json, .onnx)")
    parser.add_argument("-o1", "--output-path1", metavar="PATH1", required=False,
            help="output path1 for the optimised model (.json, .onnx)")
    parser.add_argument("-o2", "--output-path2", metavar="PATH2", required=False,
            help="output path1 for the optimised model (.json, .onnx)")
    parser.add_argument("-o3", "--output-path3", metavar="PATH3", required=False,
            help="output path3 for the optimised model (.json, .onnx)")
    parser.add_argument("--optimiser", choices=["brute", "annealing", "init", "rule"], required=False, default="annealing",
            help="optimiser to use")
    parser.add_argument('--objective', choices=['throughput','latency'], required=False, default="latency", help='Optimiser objective')
    parser.add_argument("--enable_reconf", choices=["true", "false"], required=False, default="true", help="multiple partitions")
    parser.add_argument("--enable_Huffman", choices=["true", "false"], required=False, default="true", help="Huffman Decoding")
    parser.add_argument('--seed', metavar='N', type=int, default=random.randint(0,2**32-1),
        help='Seed for the optimiser run')
    parser.add_argument('--batch-size', metavar='N', type=int, default=256,
        help='batch size')

    args = parser.parse_args(args)

    # get the batch size
    batch_size = args.batch_size
    if args.objective == "latency":
        batch_size = 1

    # print the run setting
    print("#### ( run settings ) ####")
    print(f" * model    : {args.model}")
    print(f" * separated model 1    : {args.model1}")
    print(f" * separated model 2    : {args.model2}")
    print(f" * separated model 3    : {args.model3}")
    print(f" * backend  : {args.backend}")
    print(f" * platform : {args.platform}")
    print(f" * optimiser : {args.optimiser}")
    print(f" * objective : {args.objective}")
    print(f" * batch size : {args.batch_size}")
    print(f" * enable_reconf : {args.enable_reconf}")
    print(f" * seed : {args.seed}")
    print(f" * output_path : {args.output_path}")
    print(f" * output_path : {args.output_path1}")
    print(f" * output_path : {args.output_path2}")
    print(f" * output_path : {args.output_path3}")
    print("##########################")

    # parse the platform
    with open(args.platform, "r") as f:
        platform = json.load(f)


    # get the backend parser and exporter
    parser = importlib.import_module(f"samo.backend.{args.backend}.parser")
    exporter = importlib.import_module(f"samo.backend.{args.backend}.export")

    # parse the network
    graph = parser.parse(args.model, platform, batch_size)

    graph.enable_reconf = {"true":True, "false":False}[args.enable_reconf]
    graph.objective = args.objective

    random.seed(args.seed)
    np.random.seed(args.seed)

    # init
    for partition in graph.partitions:
        partition.reset()

    # create an optimiser instance for the network
    if args.optimiser == "annealing":
        opt = SimulatedAnnealing(graph)
    elif args.optimiser == "rule":
        opt = RuleBased(graph)
        opt1 = RuleBased(graph)
        opt2 = RuleBased(graph)
        opt3 = RuleBased(graph)
    elif args.optimiser == "brute":
        opt = BruteForce(graph)
    elif args.optimiser == "init":
        graph.summary()
        exporter.export(graph, args.model, args.output_path)
        return
    else:
        raise NameError

    
    
    
    
    opt_copy = copy.deepcopy(opt)
    opt.start_time = time.time()
    
    
    # split up the network in 3 parts, assuming alll nodes are valid for split. 
    can_split = args.optimiser != "brute"
    splitted_networks = []
    while can_split:
        can_split = False
        #for i in range(len(opt.network.partitions)):
        for i in range(1):
            valid_splits = opt.network.valid_splits(i)
            network_copy = copy.deepcopy(opt.network)
            if valid_splits:
                prev = opt.network.check_constraints()#检查什么都没parttion或者优化过的partition资源够不够
                
                # # Exsuattive search 
                # splits = exsuastive_search(valid_splits, 2)#find all possiblity for splitting in 3 parts, 
                            
                # # Randomize search
                # splits = random_search(valid_splits, 2, 10) # Randomise for 8 times , split twice
               
                # Memoryfootprint search
                splits = mem_footprint_search(args,opt_copy,"BRAM_LUT_std") # strategy can be chosen form "BRAM_LUT_std" "max_BRAM" or "max_LUT"
                
                # splits = pretend_to_be_known_search(args,opt_copy)
                
                for split in splits:
                    opt.network = copy.deepcopy(network_copy)
                    opt.network.split(i, split[0])
                    opt.network.split(i+1, split[1])
                    
                    if prev and not opt.network.check_constraints():
                        can_split = False
                        opt.network = network_copy
                    else:
                        splitted_networks.append(copy.deepcopy(opt.network))
                        
    # validate generated design
    assert opt.network.check_constraints(), "Intial design infeasible!"
    
    #iterate over different splitting combination, 
    throughput_dict = {}
    time_table = {}
    # for i in range(2):
    for i in (range(len(splitted_networks))):
        opt.network = copy.deepcopy(splitted_networks[i])
        
        opt1.network= copy.deepcopy(opt.network)
        opt2.network= copy.deepcopy(opt.network)
        opt3.network= copy.deepcopy(opt.network)
        
        opt1.network.partitions.remove(opt1.network.partitions[1])
        opt1.network.partitions.remove(opt1.network.partitions[1])
        
        opt2.network.partitions.remove(opt2.network.partitions[0])
        opt2.network.partitions.remove(opt2.network.partitions[1])
        
        opt3.network.partitions.remove(opt3.network.partitions[0])
        opt3.network.partitions.remove(opt3.network.partitions[0])        
        
        # split up the network opt1,2,3  completely first to check if all nodes are valid for split
        can_split_1 = args.optimiser != "brute"
        while can_split_1:
            can_split_1 = False
            for i in range(len(opt1.network.partitions)):
                valid_splits = opt1.network.valid_splits(i)
                network_copy = copy.deepcopy(opt1.network)
                if valid_splits:
                    can_split_1 = True
                    prev = opt1.network.check_constraints_Huffman(MY_HUFFMAN_CONSTANT,MY_HUFFMAN_CONSTANT)#检查什么都没parttion或者优化过的partition资源够不够
                    opt1.network.split(i, valid_splits[0])#每次split,len(opt.network.partitions)都+1
                    if prev and not opt1.network.check_constraints_Huffman(MY_HUFFMAN_CONSTANT,MY_HUFFMAN_CONSTANT):
                        can_split_1 = False
                        opt1.network = network_copy
        
        can_split_2 = args.optimiser != "brute"
        while can_split_2:
            can_split_2 = False
            for i in range(len(opt2.network.partitions)):
                valid_splits = opt2.network.valid_splits(i)
                network_copy = copy.deepcopy(opt2.network)
                if valid_splits:
                    can_split_2 = True
                    prev = opt2.network.check_constraints_Huffman(MY_HUFFMAN_CONSTANT,MY_HUFFMAN_CONSTANT)#检查什么都没parttion或者优化过的partition资源够不够
                    opt2.network.split(i, valid_splits[0])#每次split,len(opt.network.partitions)都+1
                    if prev and not opt2.network.check_constraints_Huffman(MY_HUFFMAN_CONSTANT,MY_HUFFMAN_CONSTANT):
                        can_split_2 = False
                        opt2.network = network_copy
        
        can_split_3 = args.optimiser != "brute"
        while can_split_3:
            can_split_3 = False
            for i in range(len(opt3.network.partitions)):
                valid_splits = opt3.network.valid_splits(i)
                network_copy = copy.deepcopy(opt3.network)
                if valid_splits:
                    can_split_3 = True
                    prev = opt3.network.check_constraints_Huffman(MY_HUFFMAN_CONSTANT,MY_HUFFMAN_CONSTANT)#检查什么都没parttion或者优化过的partition资源够不够
                    opt3.network.split(i, valid_splits[0])#每次split,len(opt.network.partitions)都+1
                    if prev and not opt3.network.check_constraints_Huffman(MY_HUFFMAN_CONSTANT,MY_HUFFMAN_CONSTANT):
                        can_split_3 = False
                        opt3.network = network_copy
        
                        
        # validate generated design
        assert opt1.network.check_constraints_Huffman(MY_HUFFMAN_CONSTANT,MY_HUFFMAN_CONSTANT),"Intial design infeasible!"
        assert opt2.network.check_constraints_Huffman(MY_HUFFMAN_CONSTANT,MY_HUFFMAN_CONSTANT),"Intial design infeasible!"
        assert opt3.network.check_constraints_Huffman(MY_HUFFMAN_CONSTANT,MY_HUFFMAN_CONSTANT),"Intial design infeasible!"
        
        
        
        # run the optimiser
        opt1.optimise_multi_FPGA_Huffman(MY_HUFFMAN_CONSTANT,MY_HUFFMAN_CONSTANT)
        opt2.optimise_multi_FPGA_Huffman(MY_HUFFMAN_CONSTANT,MY_HUFFMAN_CONSTANT)
        opt3.optimise_multi_FPGA_Huffman(MY_HUFFMAN_CONSTANT,MY_HUFFMAN_CONSTANT)
       
        # validate generated design
        # assert opt1.network.check_constraints(),"Optimised design infeasible!"
        
        opt1.network.summary(1)
        opt2.network.summary(2)
        opt3.network.summary(3)
        
        networklist = (opt1.network,opt2.network,opt3.network)
        throughput1 = opt1.network.eval_throughput()
        throughput2 = opt2.network.eval_throughput()
        throughput3 = opt3.network.eval_throughput()
        throughput_list = sorted([throughput1,throughput2,throughput3]) # in series , the ones with least throughput represent the overall partition.  
        throughput_dict[throughput_list[0]] = copy.deepcopy(networklist)
        
        time_diff = time.time()-opt.start_time
        time_table[time_diff]=throughput_list[0]
        
        if opt1.network.check_constraints_multi_device(opt2.network) and \
            opt2.network.check_constraints_multi_device(opt3.network):
                throughput_dict[throughput_list[0]] = copy.deepcopy(networklist)
                ...
               
        else:
            ...
            #splitted_networks.remove(splitted_networks[i])
    
    #Sort out the network list which has max througput among all combinations.    
    sorted_components = sorted(throughput_dict.items(), key=lambda x: x[0],reverse=True)  
    # End of optimisation 
    opt.stop_time = time.time()
    time_passed = opt.stop_time-opt.start_time
    overall_throughput_improve = sorted_components[0][0] / 0.02 
    print(f"The overall throuput has increased {overall_throughput_improve:.2f} times")
    print(f"Time passed: {time_passed} s")
    # export the design
    exporter.export(sorted_components[0][1][0], args.model1, args.output_path1)
    exporter.export(sorted_components[0][1][1], args.model2, args.output_path2)
    exporter.export(sorted_components[0][1][2], args.model3, args.output_path3)
    
    
def main_for_multi(args):
    # logging configuration
    logging.basicConfig(filename='samo.log', filemode='w', level=logging.INFO)

    # parse arguments
    parser = argparse.ArgumentParser(description="SAME CNN optimiser")
    parser.add_argument("-m", "--model", metavar="PATH", required=True,
            help="path to the CNN model that you wish to optimise (.keras, .onnx)")
    parser.add_argument("-m1", "--model1", metavar="PATH1", required=True,
            help="path to the CNN model that you wish to optimise (.keras, .onnx)")
    parser.add_argument("-m2", "--model2", metavar="PATH2", required=True,
            help="path to the CNN model that you wish to optimise (.keras, .onnx)")
    parser.add_argument("-m3", "--model3", metavar="PATH3", required=True,
            help="path to the CNN model that you wish to optimise (.keras, .onnx)")
    parser.add_argument("-b", "--backend", choices=["fpgaconvnet", "finn", "hls4ml"], required=True,
            help="target backend for accelerating the model")
    parser.add_argument("-p", "--platform", metavar="PATH", required=True,
            help="hardware platform details (.json)")
    parser.add_argument("-o", "--output-path", metavar="PATH", required=True,
            help="output path for the optimised model (.json, .onnx)")
    parser.add_argument("-o1", "--output-path1", metavar="PATH1", required=False,
            help="output path1 for the optimised model (.json, .onnx)")
    parser.add_argument("-o2", "--output-path2", metavar="PATH2", required=False,
            help="output path1 for the optimised model (.json, .onnx)")
    parser.add_argument("-o3", "--output-path3", metavar="PATH3", required=False,
            help="output path3 for the optimised model (.json, .onnx)")
    parser.add_argument("--optimiser", choices=["brute", "annealing", "init", "rule"], required=False, default="annealing",
            help="optimiser to use")
    parser.add_argument('--objective', choices=['throughput','latency'], required=False, default="latency", help='Optimiser objective')
    parser.add_argument("--enable_reconf", choices=["true", "false"], required=False, default="true", help="multiple partitions")
    parser.add_argument("--enable_Huffman", choices=["true", "false"], required=False, default="true", help="Huffman Decoding")
    parser.add_argument('--seed', metavar='N', type=int, default=random.randint(0,2**32-1),
        help='Seed for the optimiser run')
    parser.add_argument('--batch-size', metavar='N', type=int, default=256,
        help='batch size')

    args = parser.parse_args(args)

    # get the batch size
    batch_size = args.batch_size
    if args.objective == "latency":
        batch_size = 1

    # print the run setting
    print("#### ( run settings ) ####")
    print(f" * model    : {args.model}")
    print(f" * separated model 1    : {args.model1}")
    print(f" * separated model 2    : {args.model2}")
    print(f" * separated model 3    : {args.model3}")
    print(f" * backend  : {args.backend}")
    print(f" * platform : {args.platform}")
    print(f" * optimiser : {args.optimiser}")
    print(f" * objective : {args.objective}")
    print(f" * batch size : {args.batch_size}")
    print(f" * enable_reconf : {args.enable_reconf}")
    print(f" * seed : {args.seed}")
    print(f" * output_path : {args.output_path}")
    print(f" * output_path : {args.output_path1}")
    print(f" * output_path : {args.output_path2}")
    print(f" * output_path : {args.output_path3}")
    print("##########################")

    # parse the platform
    with open(args.platform, "r") as f:
        platform = json.load(f)


    # get the backend parser and exporter
    parser = importlib.import_module(f"samo.backend.{args.backend}.parser")
    exporter = importlib.import_module(f"samo.backend.{args.backend}.export")

    # parse the network
    graph = parser.parse(args.model, platform, batch_size)

    graph.enable_reconf = {"true":True, "false":False}[args.enable_reconf]
    graph.objective = args.objective

    random.seed(args.seed)
    np.random.seed(args.seed)

    # init
    for partition in graph.partitions:
        partition.reset()

    # create an optimiser instance for the network
    if args.optimiser == "annealing":
        opt = SimulatedAnnealing(graph)
    elif args.optimiser == "rule":
        opt = RuleBased(graph)
        opt1 = RuleBased(graph)
        opt2 = RuleBased(graph)
        opt3 = RuleBased(graph)
    elif args.optimiser == "brute":
        opt = BruteForce(graph)
    elif args.optimiser == "init":
        graph.summary()
        exporter.export(graph, args.model, args.output_path)
        return
    else:
        raise NameError

    
    
    
    
    opt_copy = copy.deepcopy(opt)
    opt.start_time = time.time()
    
    
    # split up the network in 3 parts, assuming alll nodes are valid for split. 
    can_split = args.optimiser != "brute"
    splitted_networks = []
    while can_split:
        can_split = False
        #for i in range(len(opt.network.partitions)):
        for i in range(1):
            valid_splits = opt.network.valid_splits(i)
            network_copy = copy.deepcopy(opt.network)
            if valid_splits:
                prev = opt.network.check_constraints()#检查什么都没parttion或者优化过的partition资源够不够
                
                # # Exsuattive search 
                # splits = exsuastive_search(valid_splits, 2)#find all possiblity for splitting in 3 parts, 
                            
                # # Randomize search
                # splits = random_search(valid_splits, 2, 10) # Randomise for 8 times , split twice
               
                # Memoryfootprint search
                splits = mem_footprint_search(args,opt_copy,"BRAM_LUT_std") # strategy can be chosen form "BRAM_LUT_std" "max_BRAM" or "max_LUT"
                
                for split in splits:
                    opt.network = copy.deepcopy(network_copy)
                    opt.network.split(i, split[0])
                    opt.network.split(i+1, split[1])
                    
                    if prev and not opt.network.check_constraints():
                        can_split = False
                        opt.network = network_copy
                    else:
                        splitted_networks.append(copy.deepcopy(opt.network))
                        
    # validate generated design
    assert opt.network.check_constraints(), "Intial design infeasible!"
    
    #iterate over different splitting combination, 
    throughput_dict = {}
    time_table = {}
    # for i in range(2):
    for i in (range(len(splitted_networks))):
        opt.network = copy.deepcopy(splitted_networks[i])
        
        opt1.network= copy.deepcopy(opt.network)
        opt2.network= copy.deepcopy(opt.network)
        opt3.network= copy.deepcopy(opt.network)
        
        opt1.network.partitions.remove(opt1.network.partitions[1])
        opt1.network.partitions.remove(opt1.network.partitions[1])
        
        opt2.network.partitions.remove(opt2.network.partitions[0])
        opt2.network.partitions.remove(opt2.network.partitions[1])
        
        opt3.network.partitions.remove(opt3.network.partitions[0])
        opt3.network.partitions.remove(opt3.network.partitions[0])        
        
        # split up the network opt1,2,3  completely first to check if all nodes are valid for split
        can_split_1 = args.optimiser != "brute"
        while can_split_1:
            can_split_1 = False
            for i in range(len(opt1.network.partitions)):
                valid_splits = opt1.network.valid_splits(i)
                network_copy = copy.deepcopy(opt1.network)
                if valid_splits:
                    can_split_1 = True
                    prev = opt1.network.check_constraints()#检查什么都没parttion或者优化过的partition资源够不够
                    opt1.network.split(i, valid_splits[0])#每次split,len(opt.network.partitions)都+1
                    if prev and not opt1.network.check_constraints():
                        can_split_1 = False
                        opt1.network = network_copy
        
        can_split_2 = args.optimiser != "brute"
        while can_split_2:
            can_split_2 = False
            for i in range(len(opt2.network.partitions)):
                valid_splits = opt2.network.valid_splits(i)
                network_copy = copy.deepcopy(opt2.network)
                if valid_splits:
                    can_split_2 = True
                    prev = opt2.network.check_constraints()#检查什么都没parttion或者优化过的partition资源够不够
                    opt2.network.split(i, valid_splits[0])#每次split,len(opt.network.partitions)都+1
                    if prev and not opt2.network.check_constraints():
                        can_split_2 = False
                        opt2.network = network_copy
        
        can_split_3 = args.optimiser != "brute"
        while can_split_3:
            can_split_3 = False
            for i in range(len(opt3.network.partitions)):
                valid_splits = opt3.network.valid_splits(i)
                network_copy = copy.deepcopy(opt3.network)
                if valid_splits:
                    can_split_3 = True
                    prev = opt3.network.check_constraints()#检查什么都没parttion或者优化过的partition资源够不够
                    opt3.network.split(i, valid_splits[0])#每次split,len(opt.network.partitions)都+1
                    if prev and not opt3.network.check_constraints():
                        can_split_3 = False
                        opt3.network = network_copy
        
                        
        # validate generated design
        assert opt1.network.check_constraints(),"Intial design infeasible!"
        assert opt2.network.check_constraints(),"Intial design infeasible!"
        assert opt3.network.check_constraints(),"Intial design infeasible!"
        
        # run the optimiser
        opt1.optimise_multi_FPGA()
        opt2.optimise_multi_FPGA()
        opt3.optimise_multi_FPGA()
        
        # validate generated design
        assert opt1.network.check_constraints(),"Optimised design infeasible!"
        
        opt1.network.summary(1)
        opt2.network.summary(2)
        opt3.network.summary(3)
        
        networklist = (opt1.network,opt2.network,opt3.network)
        throughput1 = opt1.network.eval_throughput()
        throughput2 = opt2.network.eval_throughput()
        throughput3 = opt3.network.eval_throughput()
        throughput_list = sorted([throughput1,throughput2,throughput3]) # in series , the ones with least throughput represent the overall partition.  
        throughput_dict[throughput_list[0]] = copy.deepcopy(networklist)
        
        time_diff = time.time()-opt.start_time
        time_table[time_diff]=throughput_list[0]
        
        if opt1.network.check_constraints_multi_device(opt2.network) and opt2.network.check_constraints_multi_device(opt3.network):
            throughput_dict[throughput_list[0]] = copy.deepcopy(networklist)
        # else:
        #     ...
        #     #splitted_networks.remove(splitted_networks[i])
    
    #Sort out the network list which has max througput among all combinations.    
    sorted_components = sorted(throughput_dict.items(), key=lambda x: x[0],reverse=True)  
    # End of optimisation 
    opt.stop_time = time.time()
    time_passed = opt.stop_time-opt.start_time
    overall_throughput_improve = sorted_components[0][0] // 0.005 
    print(f"The overall throuput has increased {overall_throughput_improve} times")
    print(f"Time passed: {time_passed} s")
    # export the design
    exporter.export(sorted_components[0][1][0], args.model1, args.output_path1)
    exporter.export(sorted_components[0][1][1], args.model2, args.output_path2)
    exporter.export(sorted_components[0][1][2], args.model3, args.output_path3)
    
def main_for_one(args):
    
    # logging configuration
    logging.basicConfig(filename='samo.log', filemode='w', level=logging.INFO)

    # parse arguments
    parser = argparse.ArgumentParser(description="SAME CNN optimiser")
    parser.add_argument("-m", "--model", metavar="PATH", required=True,
            help="path to the CNN model that you wish to optimise (.keras, .onnx)")
    parser.add_argument("-b", "--backend", choices=["fpgaconvnet", "finn", "hls4ml"], required=True,
            help="target backend for accelerating the model")
    parser.add_argument("-p", "--platform", metavar="PATH", required=True,
            help="hardware platform details (.json)")
    parser.add_argument("-o", "--output-path", metavar="PATH", required=True,
            help="output path for the optimised model (.json, .onnx)")
    parser.add_argument("--optimiser", choices=["brute", "annealing", "init", "rule"], required=False, default="annealing",
            help="optimiser to use")
    parser.add_argument('--objective', choices=['throughput','latency'], required=False, default="latency", help='Optimiser objective')
    parser.add_argument("--enable_reconf", choices=["true", "false"], required=False, default="true", help="multiple partitions")
    parser.add_argument('--seed', metavar='N', type=int, default=random.randint(0,2**32-1),
        help='Seed for the optimiser run')
    parser.add_argument('--batch-size', metavar='N', type=int, default=256,
        help='batch size')

    args = parser.parse_args(args)

    # get the batch size
    batch_size = args.batch_size
    if args.objective == "latency":
        batch_size = 1

    # print the run setting
    print("#### ( run settings ) ####")
    print(f" * model    : {args.model}")
    print(f" * backend  : {args.backend}")
    print(f" * platform : {args.platform}")
    print(f" * optimiser : {args.optimiser}")
    print(f" * objective : {args.objective}")
    print(f" * batch size : {args.batch_size}")
    print(f" * enable_reconf : {args.enable_reconf}")
    print(f" * seed : {args.seed}")
    print(f" * output_path : {args.output_path}")
    print("##########################")

    # parse the platform
    with open(args.platform, "r") as f:
        platform = json.load(f)

    # get the backend parser and exporter
    parser = importlib.import_module(f"samo.backend.{args.backend}.parser")
    exporter = importlib.import_module(f"samo.backend.{args.backend}.export")

    # parse the network
    graph = parser.parse(args.model, platform, batch_size)

    graph.enable_reconf = {"true":True, "false":False}[args.enable_reconf]
    graph.objective = args.objective

    random.seed(args.seed)
    np.random.seed(args.seed)

    # init
    for partition in graph.partitions:
        partition.reset()

    # create an optimiser instance for the network
    if args.optimiser == "annealing":
        opt = SimulatedAnnealing(graph)
    elif args.optimiser == "rule":
        opt = RuleBased(graph)
    elif args.optimiser == "brute":
        opt = BruteForce(graph)
    elif args.optimiser == "init":
        graph.summary()
        exporter.export(graph, args.model, args.output_path)
        return
    else:
        raise NameError

    opt.start_time = time.time()

    # split up the network completely
    can_split = args.optimiser != "brute"
    while can_split:
        can_split = False
        for i in range(len(opt.network.partitions)):
            valid_splits = opt.network.valid_splits(i)
            network_copy = copy.deepcopy(opt.network)
            if valid_splits:
                can_split = True
                prev = opt.network.check_constraints()
                opt.network.split(i, valid_splits[0])
                if prev and not opt.network.check_constraints():
                    can_split = False
                    opt.network = network_copy

    # validate generated design
    assert opt.network.check_constraints(), "Intial design infeasible!"

    # run the optimiser
    opt.optimise()

    # validate generated design
    assert opt.network.check_constraints(), "Optimised design infeasible!"

    # print a summary of the run
    opt.network.summary(1)

    # export the design
    exporter.export(opt.network, args.model, args.output_path)

def main_for_one_split_Conv(args):
    
    # logging configuration
    logging.basicConfig(filename='samo.log', filemode='w', level=logging.INFO)

    # parse arguments
    parser = argparse.ArgumentParser(description="SAME CNN optimiser")
    parser.add_argument("-m", "--model", metavar="PATH", required=True,
            help="path to the CNN model that you wish to optimise (.keras, .onnx)")
    parser.add_argument("-b", "--backend", choices=["fpgaconvnet", "finn", "hls4ml"], required=True,
            help="target backend for accelerating the model")
    parser.add_argument("-p", "--platform", metavar="PATH", required=True,
            help="hardware platform details (.json)")
    parser.add_argument("-o", "--output-path", metavar="PATH", required=True,
            help="output path for the optimised model (.json, .onnx)")
    parser.add_argument("--optimiser", choices=["brute", "annealing", "init", "rule"], required=False, default="annealing",
            help="optimiser to use")
    parser.add_argument('--objective', choices=['throughput','latency'], required=False, default="latency", help='Optimiser objective')
    parser.add_argument("--enable_reconf", choices=["true", "false"], required=False, default="true", help="multiple partitions")
    parser.add_argument('--seed', metavar='N', type=int, default=random.randint(0,2**32-1),
        help='Seed for the optimiser run')
    parser.add_argument('--batch-size', metavar='N', type=int, default=256,
        help='batch size')

    args = parser.parse_args(args)

    # get the batch size
    batch_size = args.batch_size
    if args.objective == "latency":
        batch_size = 1

    # print the run setting
    print("#### ( run settings ) ####")
    print(f" * model    : {args.model}")
    print(f" * backend  : {args.backend}")
    print(f" * platform : {args.platform}")
    print(f" * optimiser : {args.optimiser}")
    print(f" * objective : {args.objective}")
    print(f" * batch size : {args.batch_size}")
    print(f" * enable_reconf : {args.enable_reconf}")
    print(f" * seed : {args.seed}")
    print(f" * output_path : {args.output_path}")
    print("##########################")

    # parse the platform
    with open(args.platform, "r") as f:
        platform = json.load(f)


    # get the backend parser and exporter
    parser = importlib.import_module(f"samo.backend.{args.backend}.parser")
    exporter = importlib.import_module(f"samo.backend.{args.backend}.export")

    # parse the network
    graph = parser.parse(args.model, platform, batch_size)

    graph.enable_reconf = {"true":True, "false":False}[args.enable_reconf]
    graph.objective = args.objective

    random.seed(args.seed)
    np.random.seed(args.seed)

    # init
    for partition in graph.partitions:
        partition.reset()

    # create an optimiser instance for the network
    if args.optimiser == "annealing":
        opt = SimulatedAnnealing(graph)
    elif args.optimiser == "rule":
        opt = RuleBased(graph) # opt 是个network instance,  
    elif args.optimiser == "brute":
        opt = BruteForce(graph)
    elif args.optimiser == "init":
        graph.summary()
        exporter.export(graph, args.model, args.output_path)
        return
    else:
        raise NameError

    opt.start_time = time.time()

    # split up the network completely
    can_split = args.optimiser != "brute"
    while can_split:
        can_split = False
        for i in range(len(opt.network.partitions)):
            valid_splits = opt.network.valid_splits(i)
            network_copy = copy.deepcopy(opt.network)
            if valid_splits:
                can_split = True
                prev = opt.network.check_constraints()
                opt.network.split(i, valid_splits[0])
                if prev and not opt.network.check_constraints():
                    can_split = False
                    opt.network = network_copy


    
    # Step 1: Identify the partition
    partition_to_split = opt.network.partitions[0]  # Get the partition 
    # Step 2: Get the Convolution Layer object from the node
    for layer in partition_to_split.nodes():
        conv_layer = partition_to_split.nodes[layer]["hw"]
        # if (opt.network.get_layer_type_from_node(partition_to_split.graph, node) == LAYER_TYPE.Convolution):
            # conv_layer_node = partition_to_split.nodes[0]
         # Get the Convolution Layer node from the partition

    # Step 3: Create two separate Convolution Layer objects based on the original Convolution Layer
    conv_layer_split1, conv_layer_split2 = opt.network.get_seperate_layer_hardware(conv_layer)

    # Step 4: Add the two new Convolution Layer nodes to the partition's graph
    new_conv_node1 = "new_conv_node1"  # Unique name for the first new Convolution Layer node
    new_conv_node2 = "new_conv_node2"  # Unique name for the second new Convolution Layer node

    partition_to_split.graph.add_node(new_conv_node1, type=LAYER_TYPE.Convolution, hw=conv_layer_split1, inputs={})
    partition_to_split.graph.add_node(new_conv_node2, type=LAYER_TYPE.Convolution, hw=conv_layer_split2, inputs={})

    # Step 5: Modify the connections in the graph to connect the new Convolution Layer nodes
    # Connect the input nodes to new_conv_node1 and new_conv_node2
    for input_node in partition_to_split.graph.predecessors(conv_layer_node):
        partition_to_split.graph.add_edge(input_node, new_conv_node1)

    for input_node in partition_to_split.graph.predecessors(conv_layer_node):
        partition_to_split.graph.add_edge(input_node, new_conv_node2)

    # Connect new_conv_node1 and new_conv_node2 to the output nodes
    for output_node in partition_to_split.graph.successors(conv_layer_node):
        partition_to_split.graph.add_edge(new_conv_node1, output_node)

    for output_node in partition_to_split.graph.successors(conv_layer_node):
        partition_to_split.graph.add_edge(new_conv_node2, output_node)

    # Remove the original Convolution Layer node from the graph
    partition_to_split.graph.remove_node(conv_layer_node)

    # Step 6: Update any relevant attributes or parameters of the partition
    # For example, you may need to update wr_factor or wr_layer if applicable
    partition_to_split.wr_factor = ...  # Update wr_factor if needed
    partition_to_split.wr_layer = ...   # Update wr_layer if needed

                
    # validate generated design
    assert opt.network.check_constraints(), "Intial design infeasible!"

    # run the optimiser
    opt.optimise()

    # validate generated design
    assert opt.network.check_constraints(), "Optimised design infeasible!"

    # print a summary of the run
    opt.network.summary(1)

    # export the design
    exporter.export(opt.network, args.model, args.output_path)


if __name__ == "__main__":
    main(sys.argv[1:])

def split_and_optimise (opt1,opt2,opt3,args):
        
        # split up the network opt1,2,3  completely first to check if all nodes are valid for split
        can_split_1 = args.optimiser != "brute"
        while can_split_1:
            can_split_1 = False
            for i in range(len(opt1.network.partitions)):
                valid_splits = opt1.network.valid_splits(i)
                network_copy = copy.deepcopy(opt1.network)
                if valid_splits:
                    can_split_1 = True
                    prev = opt1.network.check_constraints()#检查什么都没parttion或者优化过的partition资源够不够
                    opt1.network.split(i, valid_splits[0])#每次split,len(opt.network.partitions)都+1
                    if prev and not opt1.network.check_constraints():
                        can_split_1 = False
                        opt1.network = network_copy
        
        can_split_2 = args.optimiser != "brute"
        while can_split_2:
            can_split_2 = False
            for i in range(len(opt2.network.partitions)):
                valid_splits = opt2.network.valid_splits(i)
                network_copy = copy.deepcopy(opt2.network)
                if valid_splits:
                    can_split_2 = True
                    prev = opt2.network.check_constraints()#检查什么都没parttion或者优化过的partition资源够不够
                    opt2.network.split(i, valid_splits[0])#每次split,len(opt.network.partitions)都+1
                    if prev and not opt2.network.check_constraints():
                        can_split_2 = False
                        opt2.network = network_copy
        
        can_split_3 = args.optimiser != "brute"
        while can_split_3:
            can_split_3 = False
            for i in range(len(opt3.network.partitions)):
                valid_splits = opt3.network.valid_splits(i)
                network_copy = copy.deepcopy(opt3.network)
                if valid_splits:
                    can_split_3 = True
                    prev = opt3.network.check_constraints()#检查什么都没parttion或者优化过的partition资源够不够
                    opt3.network.split(i, valid_splits[0])#每次split,len(opt.network.partitions)都+1
                    if prev and not opt3.network.check_constraints():
                        can_split_3 = False
                        opt3.network = network_copy
        
                        
        # validate generated design
        assert opt1.network.check_constraints(),"Intial design infeasible!"
        assert opt2.network.check_constraints(),"Intial design infeasible!"
        assert opt3.network.check_constraints(),"Intial design infeasible!"
        
        # run the optimiser
        opt1.optimise_multi_FPGA()
        opt2.optimise_multi_FPGA()
        opt3.optimise_multi_FPGA()
        
        assert opt1.network.check_constraints(),"Optimised design infeasible!"
        
        opt1.network.summary(1)
        opt2.network.summary(2)
        opt3.network.summary(3)
        
        # validate generated design
        if opt1.network.check_constraints_multi_device(opt2.network):
            #assemble them back together to cculate the expected total throuput. 
            opt.network.partitions.remove(opt.network.partitions[0])  
            opt.network.partitions.remove(opt.network.partitions[0])  
            opt.network.partitions.remove(opt.network.partitions[0])
            
            opt.network.partitions.append(opt1.network.partitions[0])
            opt.network.partitions.append(opt2.network.partitions[0])
            opt.network.partitions.append(opt3.network.partitions[0])
                        
            # print a summary of the run
            opt.network.summary()
            throughput = opt.network.eval_throughput()
            network_list = []
            network_list.append(opt1.network)
            network_list.append(opt2.network)
            network_list.append(opt3.network)
            throughput_dict[throughput] = copy.deepcopy(network_list)
        else:
            splitted_networks.remove(splitted_networks[i])
            
        #Sort out the network list which has max througput among all combination.    
        sorted_components = sorted(throughput_dict.items(), key=lambda x: x[0])
        
def exsuastive_search(split_list, split_number):
    return list(itertools.combinations(split_list, split_number))

def random_search(split_list, split_number, radnomise_iteration_number):
    mid_list = []
    all_uniqe = False
    split_list_copy = copy.deepcopy(split_list)   
    while not all_uniqe:
        mid_list.append(random.sample(split_list_copy, split_number))
        mid_list[-1].sort(key=lambda x: split_list_copy.index(x))
        mid_list = remove_duplicates_preserve_order(mid_list)
        
        if len(mid_list) == radnomise_iteration_number:
            all_uniqe = True
        
        
    return mid_list  

def mem_footprint_search(args,opt,strategy):
    # split up the network completely
    can_split = args.optimiser != "brute"
    split_list = opt.network.valid_splits(0)
    while can_split:
        can_split = False
        for i in range(len(opt.network.partitions)):
            valid_splits = opt.network.valid_splits(i)
            network_copy = copy.deepcopy(opt.network)
            if valid_splits:
                can_split = True
                prev = opt.network.check_constraints()
                opt.network.split(i, valid_splits[0])
                if prev and not opt.network.check_constraints():
                    can_split = False
                    opt.network = network_copy
    
    # validate generated design
    assert opt.network.check_constraints(), "Intial mem footprint mapping infeasible!"
    partition_mem_footprint = []
    for partition in opt.network.partitions:
        partition_mem_footprint.append(partition.eval_resource()) 
    
    mid_list = []
    num_partitions = 3
    goal_list = ["max_BRAM","max_LUT","BRAM_LUT_std"]
    for goal in goal_list:
        partition1_idx, partition2_idx = partition_network(partition_mem_footprint, goal)
        mid_list.append([split_list[partition1_idx-1],split_list[partition2_idx-1]])
    return mid_list

def pretend_to_be_known_search(args,opt):
    # split up the network completely
    can_split = args.optimiser != "brute"
    split_list = opt.network.valid_splits(0)
    while can_split:
        can_split = False
        for i in range(len(opt.network.partitions)):
            valid_splits = opt.network.valid_splits(i)
            network_copy = copy.deepcopy(opt.network)
            if valid_splits:
                can_split = True
                prev = opt.network.check_constraints()
                opt.network.split(i, valid_splits[0])
                if prev and not opt.network.check_constraints():
                    can_split = False
                    opt.network = network_copy
    
    # validate generated design
    assert opt.network.check_constraints(), "Intial mem footprint mapping infeasible!"
    partition_mem_footprint = []
    for partition in opt.network.partitions:
        partition_mem_footprint.append(partition.eval_resource()) 
    
    mid_list = []
    mid_list.append([split_list[1],split_list[3]])
    return mid_list
    
def remove_duplicates_preserve_order(lst):
    unique_list = []
    for item in lst:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list
   
def partition_network(memory_footprints,goal):# For now 固定3个prtition, goal chose between "max_BRAM" "max_LUT" and "BRAM_LUT_std"
    # Initialize variables to store the best partition indices and the total memory for each partition
    best_partition_indices = []
    best_partition_memory_BRAM = float('inf')
    best_partition_memory_LUT = float('inf')
    best_partition_memory_type2 = float('inf')
    min_std_deviation = float('inf')
    # Calculate the total memory of the network for each resource (LUT and BRAM)
    total_memory_LUT = sum(layer["LUT"] for layer in memory_footprints)
    total_memory_BRAM = sum(layer["BRAM"] for layer in memory_footprints)

    # Iterate through possible partition indices
    for idx1 in range(1, len(memory_footprints) - 1):
        for idx2 in range(idx1 + 1, len(memory_footprints)):
            # Calculate the memory of each partition for both resources (LUT and BRAM)
            partition1_memory_LUT = sum(layer["LUT"] for layer in memory_footprints[:idx1])
            partition1_memory_BRAM = sum(layer["BRAM"] for layer in memory_footprints[:idx1])

            partition2_memory_LUT = sum(layer["LUT"] for layer in memory_footprints[idx1:idx2])
            partition2_memory_BRAM = sum(layer["BRAM"] for layer in memory_footprints[idx1:idx2])

            partition3_memory_LUT = sum(layer["LUT"] for layer in memory_footprints[idx2:])
            partition3_memory_BRAM = sum(layer["BRAM"] for layer in memory_footprints[idx2:])

            # Calculate the maximum memory for both resources among the three partitions
            max_partition_memory_LUT = max(partition1_memory_LUT, partition2_memory_LUT, partition3_memory_LUT)
            max_partition_memory_BRAM = max(partition1_memory_BRAM, partition2_memory_BRAM, partition3_memory_BRAM)

            # Calculate a combined score considering both resources ( adjust this scoring function based on requirements)
            # 或者Combined_score = max_partition_memory_LUT + max_partition_memory_BRAM
            # 或者Combined_score = max_partition_memory_LUT + max_partition_memory_BRAM + max_DSP
            # 或者先满足BRAM之后,再满足LUT, 再满足DSP...
            if goal == "max_BRAM":
                combined_score = max_partition_memory_BRAM
                best_partition_memory_type = best_partition_memory_BRAM
                max_partition_memory_type2 = max_partition_memory_LUT
                
                # Update the best partition indices if a better configuration is found
                if (combined_score < best_partition_memory_type) or ((combined_score == best_partition_memory_BRAM) and (max_partition_memory_LUT< best_partition_memory_LUT)):
                    best_partition_indices = [idx1, idx2]
                    best_partition_memory_type = combined_score
                    best_partition_memory_BRAM = max_partition_memory_BRAM
                    best_partition_memory_LUT = max_partition_memory_LUT
                    if max_partition_memory_type2 < best_partition_memory_type2:
                        best_partition_memory_type2 = max_partition_memory_type2  
                # # Update the best partition indices if a better configuration is found
                # if (combined_score < best_partition_memory_BRAM) or ((combined_score == best_partition_memory_BRAM) and (max_partition_memory_LUT< best_partition_memory_LUT)):
                #     best_partition_indices = [idx1, idx2]
                #     best_partition_memory_BRAM = combined_score
                #     if max_partition_memory_LUT < best_partition_memory_LUT:
                #         best_partition_memory_LUT = max_partition_memory_LUT            
            elif goal == "max_LUT":
                combined_score = max_partition_memory_LUT
                best_partition_memory_type = best_partition_memory_LUT
                max_partition_memory_type2 = max_partition_memory_BRAM
                # Update the best partition indices if a better configuration is found
                if (combined_score < best_partition_memory_type) or ((combined_score == best_partition_memory_BRAM) and (max_partition_memory_LUT< best_partition_memory_LUT)):
                    best_partition_indices = [idx1, idx2]
                    best_partition_memory_type = combined_score
                    best_partition_memory_BRAM = max_partition_memory_BRAM
                    best_partition_memory_LUT = max_partition_memory_LUT
                    if max_partition_memory_type2 < best_partition_memory_type2:
                        best_partition_memory_type2 = max_partition_memory_type2  
                
            elif goal == "LUT+BRAM":
                combined_score =  max_partition_memory_LUT + max_partition_memory_BRAM   
            
            elif goal == "BRAM_LUT_std":
                # Calculate the standard deviation of BRAM memory among the three partitions
                BRAM_memory_list = [partition1_memory_BRAM, partition2_memory_BRAM, partition3_memory_BRAM]
                LUT_memory_list = [ partition1_memory_LUT, partition2_memory_LUT, partition3_memory_LUT]
                std_deviation = statistics.stdev(BRAM_memory_list) + statistics.stdev(LUT_memory_list)

                # Update the best partition indices if a configuration with lower standard deviation is found
                if std_deviation < min_std_deviation:
                    best_partition_indices = [idx1, idx2]
                    min_std_deviation = std_deviation

              
            

    # Calculate the indices of the partitions in the original list
    partition1_idx, partition2_idx, partition3_idx = best_partition_indices[0], best_partition_indices[1], len(memory_footprints)

    return partition1_idx, partition2_idx    

def main(args):
    
    # logging configuration
    logging.basicConfig(filename='samo.log', filemode='w', level=logging.INFO)

    # parse arguments
    parser = argparse.ArgumentParser(description="SAME CNN optimiser")
    parser.add_argument("-m", "--model", metavar="PATH", required=True,
            help="path to the CNN model that you wish to optimise (.keras, .onnx)")
    parser.add_argument("-b", "--backend", choices=["fpgaconvnet", "finn", "hls4ml"], required=True,
            help="target backend for accelerating the model")
    parser.add_argument("-p", "--platform", metavar="PATH", required=True,
            help="hardware platform details (.json)")
    parser.add_argument("-o", "--output-path", metavar="PATH", required=True,
            help="output path for the optimised model (.json, .onnx)")
    parser.add_argument("--optimiser", choices=["brute", "annealing", "init", "rule"], required=False, default="annealing",
            help="optimiser to use")
    parser.add_argument('--objective', choices=['throughput','latency'], required=False, default="latency", help='Optimiser objective')
    parser.add_argument("--enable_reconf", choices=["true", "false"], required=False, default="true", help="multiple partitions")
    parser.add_argument('--seed', metavar='N', type=int, default=random.randint(0,2**32-1),
        help='Seed for the optimiser run')
    parser.add_argument('--batch-size', metavar='N', type=int, default=256,
        help='batch size')

    args = parser.parse_args(args)

    # get the batch size
    batch_size = args.batch_size
    if args.objective == "latency":
        batch_size = 1

    # print the run setting
    print("#### ( run settings ) ####")
    print(f" * model    : {args.model}")
    print(f" * backend  : {args.backend}")
    print(f" * platform : {args.platform}")
    print(f" * optimiser : {args.optimiser}")
    print(f" * objective : {args.objective}")
    print(f" * batch size : {args.batch_size}")
    print(f" * enable_reconf : {args.enable_reconf}")
    print(f" * seed : {args.seed}")
    print(f" * output_path : {args.output_path}")
    print("##########################")

    # parse the platform
    with open(args.platform, "r") as f:
        platform = json.load(f)


    # get the backend parser and exporter
    parser = importlib.import_module(f"samo.backend.{args.backend}.parser")
    exporter = importlib.import_module(f"samo.backend.{args.backend}.export")

    # parse the network
    graph = parser.parse(args.model, platform, batch_size)

    graph.enable_reconf = {"true":True, "false":False}[args.enable_reconf]
    graph.objective = args.objective

    random.seed(args.seed)
    np.random.seed(args.seed)

    # init
    for partition in graph.partitions:
        partition.reset()

    # create an optimiser instance for the network
    if args.optimiser == "annealing":
        opt = SimulatedAnnealing(graph)
    elif args.optimiser == "rule":
        opt = RuleBased(graph)
    elif args.optimiser == "brute":
        opt = BruteForce(graph)
    elif args.optimiser == "init":
        graph.summary()
        exporter.export(graph, args.model, args.output_path)
        return
    else:
        raise NameError

    opt.start_time = time.time()

    # split up the network completely
    can_split = args.optimiser != "brute"
    while can_split:
        can_split = False
        for i in range(len(opt.network.partitions)):
            valid_splits = opt.network.valid_splits(i)
            network_copy = copy.deepcopy(opt.network)
            if valid_splits:
                can_split = True
                prev = opt.network.check_constraints()
                opt.network.split(i, valid_splits[0])
                if prev and not opt.network.check_constraints():
                    can_split = False
                    opt.network = network_copy

    # validate generated design
    assert opt.network.check_constraints(), "Intial design infeasible!"

    # run the optimiser
    opt.optimise()

    # validate generated design
    assert opt.network.check_constraints(), "Optimised design infeasible!"

    # print a summary of the run
    opt.network.summary()

    # export the design
    exporter.export(opt.network, args.model, args.output_path)
