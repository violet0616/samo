import logging
import csv
import copy
import itertools
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm
import time
import os

import sys
sys.path.append('C:\\IC\\2023_project\\2023_project\\FpgaConvnet_forked\\fpgaconvnet-tutorial\\samo\\samo')  # Add the path to the folder containing the cli script

from samo.model import Network

@dataclass
class RuleBased:
    network: Network

    def update(self):
        for partition in self.network.partitions:
            for layer in partition:
                partition.nodes[layer]["hw"].update(hw_update=True)

    def optimise_single_partition(self, partition_index):
        if not self.network.partitions[partition_index].check_constraints():
            return False

        
        all_config_candidate = []
        
        step = True
        while step: #  while loop for iteratin each possible combination of configurartion, exclude the ones that has less total product in each loop
            step = False
            partition = self.network.partitions[partition_index]

            node_latencys = np.array([
                partition.nodes[layer]["hw"].latency() for layer in list(partition.nodes())])

            node_index = np.argsort(node_latencys)[-1]
            layer = list(partition.nodes())[node_index]
            node_hw = partition.nodes[layer]["hw"]

            layer_configurations = list(itertools.product(
                node_hw.valid_channel_in_folding,
                node_hw.valid_channel_out_folding,
                node_hw.valid_kernel_folding))

            current_config = [node_hw.channel_in_folding,
                    node_hw.channel_out_folding, node_hw.kernel_folding]

            layer_configurations = list(filter(
                lambda x: np.prod(x) > np.prod(current_config), layer_configurations)) # Filter Out the configurartion that has less kernal size and folding in/out channel, becasue there surly have less efficiency

            if node_hw.constraints["matching_intra_folding"]:
                layer_configurations = list(filter(lambda x: x[0] == x[1], layer_configurations))

            layer_configurations = sorted(layer_configurations, key=lambda x: np.prod(x))

            # uncomment the following code, faster optimiser but worse performance
            # def leq_folding(config):
            #    for i in range(len(config)):
            #        if config[i] < current_config[i]:
            #            return False
            #    return True
            # layer_configurations = list(filter(leq_folding, layer_configurations))

            if len(layer_configurations) > 0:# r如符合条件的congigurartion 个数大于0；
                step_candidates = {}
                next_folding_candidates = []

                prev_throughput_in = partition.eval_throughput_in()
                prev_throughput_out = partition.eval_throughput_out()
                try_merge_prev = False
                try_merge_next = False

                # iterate over configurations
                for config in layer_configurations: #  for loop for trying nect candicate, if that one is within the limit, than chose this one and save the time to try in the outter while loop 

                    # only explore the next and closest folding
                    if len(next_folding_candidates) > 1:
                        break

                    # get the partition
                    partition = self.network.partitions[partition_index]

                    # get the hardware for the layer
                    network_copy = copy.deepcopy(self.network)
                    node_hw = partition.nodes[layer]["hw"]

                    # update input channel folding
                    logging.info(f"({layer}) input channel folding = {config[0]}")
                    node_hw.channel_in_folding = config[0]
                    partition.folding_match(layer, config[0], "io")

                    # update output channel folding
                    logging.info(f"({layer}) output channel folding = {config[1]}")
                    node_hw.channel_out_folding = config[1]
                    partition.folding_match(layer, config[1], "io")

                    # update output channel folding
                    logging.info(f"({layer}) kernel folding = {config[2]}")
                    node_hw.kernel_folding = config[2]

                    # update the network
                    self.update()

                    # check the network is within platform resource constraints
                    if self.network.check_constraints():#如果resource 够用则整个network存在dictionary里
                        step_candidates[config] = copy.deepcopy(self.network)
                        next_folding_candidates.append(np.prod(config))
                        next_folding_candidates = list(set(next_folding_candidates))
                    elif not partition.check_memory_bandwdith_constraint(): # resource不够用且超了Mem_BW 
                        curr_throughput_in = partition.eval_throughput_in()
                        curr_throughput_out = partition.eval_throughput_out()

                        if curr_throughput_in > prev_throughput_in:
                            try_merge_prev = True
                        if curr_throughput_out > prev_throughput_out:
                            try_merge_next = True

                    self.network = network_copy

                step = len(step_candidates) > 0

                # choose the transformation with minimal resource
                minimal_candidate = list(sorted(step_candidates.items(),#.items returns a list of "key-value" pairs 
                    key=lambda kv: kv[1].partitions[partition_index].avg_rsc_util())) #kv[1] corresponds to the value in dictionary and kv[0] corresponds to the key

                # if a minimal candidate exists, update the network
                if minimal_candidate != []:
                    self.network = minimal_candidate[0][1]
                    # store this config with min resource to list
                    all_config_candidate.append(minimal_candidate[0])#later如果资源超了，可以回来试试第二的configurration
                
                
                    
            else:
                
                try_merge_prev = True
                try_merge_next = True

        partition = self.network.partitions[partition_index]
        if partition.eval_latency()/partition.freq < partition.platform["reconf_time"]:
            partition.try_merge_prev = True
            partition.try_merge_next = True
        else:
            partition.try_merge_prev = try_merge_prev
            partition.try_merge_next = try_merge_next

        #self.network.summary()

        return True

    def optimise_single_partition_Huffman(self, partition_index, Huffmanencode_rate, Huffmandecode_rate):
        if not self.network.partitions[partition_index].check_constraints():
            return False

        
        all_config_candidate = []
        
        step = True
        while step: #  while loop for iteratin each possible combination of configurartion, exclude the ones that has less total product in each loop
            step = False
            partition = self.network.partitions[partition_index]

            node_latencys = np.array([
                partition.nodes[layer]["hw"].latency() for layer in list(partition.nodes())])

            node_index = np.argsort(node_latencys)[-1]
            layer = list(partition.nodes())[node_index]
            node_hw = partition.nodes[layer]["hw"]

            layer_configurations = list(itertools.product(
                node_hw.valid_channel_in_folding,
                node_hw.valid_channel_out_folding,
                node_hw.valid_kernel_folding))

            current_config = [node_hw.channel_in_folding,
                    node_hw.channel_out_folding, node_hw.kernel_folding]

            layer_configurations = list(filter(
                lambda x: np.prod(x) > np.prod(current_config), layer_configurations)) # Filter Out the configurartion that has less kernal size and folding in/out channel, becasue there surly have less efficiency

            if node_hw.constraints["matching_intra_folding"]:
                layer_configurations = list(filter(lambda x: x[0] == x[1], layer_configurations))

            layer_configurations = sorted(layer_configurations, key=lambda x: np.prod(x))

            # uncomment the following code, faster optimiser but worse performance
            # def leq_folding(config):
            #    for i in range(len(config)):
            #        if config[i] < current_config[i]:
            #            return False
            #    return True
            # layer_configurations = list(filter(leq_folding, layer_configurations))

            if len(layer_configurations) > 0:# r如符合条件的congiguration 个数大于0；
                step_candidates = {}
                next_folding_candidates = []

                prev_throughput_in = partition.eval_throughput_in()
                prev_throughput_out = partition.eval_throughput_out()
                try_merge_prev = False
                try_merge_next = False

                # iterate over configurations
                for config in layer_configurations: #  for loop for trying next candicate, if that one is within the limit, than chose this one and save the time to try in the outter while loop 

                    # only explore the next and closest folding
                    if len(next_folding_candidates) > 1:
                        break

                    # get the partition
                    partition = self.network.partitions[partition_index]

                    # get the hardware for the layer
                    network_copy = copy.deepcopy(self.network)
                    node_hw = partition.nodes[layer]["hw"]

                    # update input channel folding
                    logging.info(f"({layer}) input channel folding = {config[0]}")
                    node_hw.channel_in_folding = config[0]
                    partition.folding_match(layer, config[0], "io")

                    # update output channel folding
                    logging.info(f"({layer}) output channel folding = {config[1]}")
                    node_hw.channel_out_folding = config[1]
                    partition.folding_match(layer, config[1], "io")

                    # update output channel folding
                    logging.info(f"({layer}) kernel folding = {config[2]}")
                    node_hw.kernel_folding = config[2]

                    # update the network
                    self.update()

                    # check the network is within platform resource constraints
                    if self.network.check_constraints_Huffman(Huffmanencode_rate, Huffmandecode_rate):#如果resource 够用则整个network存在dictionary里
                        step_candidates[config] = copy.deepcopy(self.network)
                        next_folding_candidates.append(np.prod(config))
                        next_folding_candidates = list(set(next_folding_candidates))
                    elif not partition.check_memory_bandwdith_constraint_Huffman(Huffmanencode_rate, Huffmandecode_rate): # resource不够用且超了Mem_BW 
                        curr_throughput_in = partition.eval_throughput_in()
                        curr_throughput_out = partition.eval_throughput_out()

                        if curr_throughput_in > prev_throughput_in:
                            try_merge_prev = True
                        if curr_throughput_out > prev_throughput_out:
                            try_merge_next = True

                    self.network = network_copy

                step = len(step_candidates) > 0

                # choose the transformation with minimal resource
                minimal_candidate = list(sorted(step_candidates.items(),#.items returns a list of "key-value" pairs 
                    key=lambda kv: kv[1].partitions[partition_index].avg_rsc_util())) #kv[1] corresponds to the value in dictionary and kv[0] corresponds to the key

                # if a minimal candidate exists, update the network
                if minimal_candidate != []:
                    self.network = minimal_candidate[0][1]
                    # store this config with min resource to list
                    all_config_candidate.append(minimal_candidate[0])#later如果资源超了，可以回来试试第二的configurration
                
                
                    
            else:
                
                try_merge_prev = True
                try_merge_next = True

        partition = self.network.partitions[partition_index]
        if partition.eval_latency()/partition.freq < partition.platform["reconf_time"]:
            partition.try_merge_prev = True
            partition.try_merge_next = True
        else:
            partition.try_merge_prev = try_merge_prev
            partition.try_merge_next = try_merge_next

        #self.network.summary()

        return True

    def optimise_single_partition_multi_device(self, partition_index):
        if not self.network.partitions[partition_index].check_constraints():
            return False

        step = True
        while step:
            step = False
            partition = self.network.partitions[partition_index]

            node_latencys = np.array([
                partition.nodes[layer]["hw"].latency() for layer in list(partition.nodes())])

            node_index = np.argsort(node_latencys)[-1]
            layer = list(partition.nodes())[node_index]
            node_hw = partition.nodes[layer]["hw"]

            layer_configurations = list(itertools.product(
                node_hw.valid_channel_in_folding,
                node_hw.valid_channel_out_folding,
                node_hw.valid_kernel_folding))

            current_config = [node_hw.channel_in_folding,
                    node_hw.channel_out_folding, node_hw.kernel_folding]

            layer_configurations = list(filter(
                lambda x: np.prod(x) > np.prod(current_config), layer_configurations))

            if node_hw.constraints["matching_intra_folding"]:
                layer_configurations = list(filter(lambda x: x[0] == x[1], layer_configurations))

            layer_configurations = sorted(layer_configurations, key=lambda x: np.prod(x))

            # uncomment the following code, faster optimiser but worse performance
            #def leq_folding(config):
            #    for i in range(len(config)):
            #        if config[i] < current_config[i]:
            #            return False
            #    return True
            #layer_configurations = list(filter(leq_folding, layer_configurations))

            if len(layer_configurations) > 0:# r如符合条件的congigurartion 个数大于0；
                step_candidates = {}
                next_folding_candidates = []

                prev_throughput_in = partition.eval_throughput_in()
                prev_throughput_out = partition.eval_throughput_out()
                try_merge_prev = False
                try_merge_next = False

                # iterate over configurations
                for config in layer_configurations:

                    # only explore the next and closest folding
                    if len(next_folding_candidates) > 1:
                        break

                    # get the partition
                    partition = self.network.partitions[partition_index]

                    # get the hardware for the layer
                    network_copy = copy.deepcopy(self.network)
                    node_hw = partition.nodes[layer]["hw"]

                    # update input channel folding
                    logging.info(f"({layer}) input channel folding = {config[0]}")
                    node_hw.channel_in_folding = config[0]
                    partition.folding_match(layer, config[0], "io")

                    # update output channel folding
                    logging.info(f"({layer}) output channel folding = {config[1]}")
                    node_hw.channel_out_folding = config[1]
                    partition.folding_match(layer, config[1], "io")

                    # update output channel folding
                    logging.info(f"({layer}) kernel folding = {config[2]}")
                    node_hw.kernel_folding = config[2]

                    # update the network
                    self.update()

                    # check the network is within platform resource constraints
                    if self.network.check_constraints() and partition.check_memory_bandwdith_constraint():
                        
                        step_candidates[config] = copy.deepcopy(self.network)
                    #     next_folding_candidates.append(np.prod(config))
                    #     next_folding_candidates = list(set(next_folding_candidates))
                    # elif not partition.check_memory_bandwdith_constraint():
                    #     curr_throughput_in = partition.eval_throughput_in()
                    #     curr_throughput_out = partition.eval_throughput_out()

                    #     if curr_throughput_in > prev_throughput_in:
                    #         try_merge_prev = True
                    #     if curr_throughput_out > prev_throughput_out:
                    #         try_merge_next = True

                    self.network = network_copy

                step = len(step_candidates) > 0

                # choose the transformation with minimal resource
                minimal_candidate = list(sorted(step_candidates.items(),
                    key=lambda kv: kv[1].partitions[partition_index].avg_rsc_util()))

                # if a minimal candidate exists, update the network
                if minimal_candidate != []:
                    self.network = minimal_candidate[0][1]

            else:
                try_merge_prev = True
                try_merge_next = True

        partition = self.network.partitions[partition_index]
        if partition.eval_latency()/partition.freq < partition.platform["reconf_time"]:
            partition.try_merge_prev = True
            partition.try_merge_next = True
        else:
            partition.try_merge_prev = try_merge_prev
            partition.try_merge_next = try_merge_next

        # partition.summary()

        return True

    def merge_partitions(self):
        # print("resolving memory bound partitions")
        reject_list = []
        i =0 

        while True:
            partitions = copy.deepcopy(self.network.partitions)
            cost = self.network.eval_cost()

            merge_prev_candidates = []
            merge_next_candidates = []
            for partition_index, partition in enumerate(self.network.partitions):# 一个一个assingning elmenten to "partition" and index to "partition_index"
                if partition_index != 0 and partition.try_merge_prev and \
                        (partition_index-1, partition_index) not in reject_list:
                    merge_prev_candidates.append(partition_index)
                    
                if partition_index != len(self.network.partitions)-1 and \
                        partition.try_merge_next and \
                        (partition_index, partition_index+1) not in reject_list:
                    merge_next_candidates.append(partition_index)
            merge_total_candidates = merge_prev_candidates + merge_next_candidates
            merge_total_candidates = list(set(merge_total_candidates))

            if len(merge_total_candidates) == 0:
                break

            partition_latencys = [
                    self.network.partitions[partition_index].eval_latency() \
                            for partition_index in merge_total_candidates]

            partition_index = merge_total_candidates[
                    partition_latencys.index(max(partition_latencys))]      #By choosing the partition with the maximum evaluation latency, it prioritizes merging partitions that might benefit from consolidation to improve performance or reduce latency.

            # merge current partition with next partition
            if partition_index in merge_next_candidates:                          # WHY??????????????????????????????????????????????
                merge_pair = (partition_index, partition_index+1)
            # merge current partition with previous partition
            elif partition_index in merge_prev_candidates:
                merge_pair = (partition_index-1, partition_index)           #即在prev 也在next 怎么办，直接merge to pre???????????????????????

            # reset both partitions to a minimal state
            self.network.partitions[merge_pair[0]].reset()
            self.network.partitions[merge_pair[1]].reset()

            # merge partitions
            self.network.merge(merge_pair)

            # optimise the new partition
            status = self.optimise_single_partition(merge_pair[0])

            # only keep if it can merge, and the performance is better
            if not status or self.network.eval_cost() >= cost:
                self.network.partitions = partitions
                reject_list.append(merge_pair)
                logging.info(f"merging {merge_pair[0]} with {merge_pair[1]} rejected")
                print("merging not accepted")
            else:
                for i, merge in enumerate(reject_list):
                    if merge[0] >= merge_pair[1]:
                        reject_list[i] = (merge[0]-1,merge[1]-1)
                logging.info(f"merging {merge_pair[0]} with {merge_pair[1]} accepted")
                print("merging accepted")

            
            # increment while loop iterration index 
            i+=1
    
    def merge_partitions_Huffman(self,Huffmanencode_rate, Huffmandecode_rate):
        # print("resolving memory bound partitions")
        reject_list = []
        i =0 

        while True:
            partitions = copy.deepcopy(self.network.partitions)
            cost = self.network.eval_cost()

            merge_prev_candidates = []
            merge_next_candidates = []
            for partition_index, partition in enumerate(self.network.partitions):# 一个一个assingning elmenten to "partition" and index to "partition_index"
                if partition_index != 0 and partition.try_merge_prev and \
                        (partition_index-1, partition_index) not in reject_list:
                    merge_prev_candidates.append(partition_index)
                    
                if partition_index != len(self.network.partitions)-1 and \
                        partition.try_merge_next and \
                        (partition_index, partition_index+1) not in reject_list:
                    merge_next_candidates.append(partition_index)
            merge_total_candidates = merge_prev_candidates + merge_next_candidates
            merge_total_candidates = list(set(merge_total_candidates))

            if len(merge_total_candidates) == 0:
                break

            partition_latencys = [
                    self.network.partitions[partition_index].eval_latency() \
                            for partition_index in merge_total_candidates]

            partition_index = merge_total_candidates[
                    partition_latencys.index(max(partition_latencys))]      #By choosing the partition with the maximum evaluation latency, it prioritizes merging partitions that might benefit from consolidation to improve performance or reduce latency.

            # merge current partition with next partition
            if partition_index in merge_next_candidates:                          # WHY??????????????????????????????????????????????
                merge_pair = (partition_index, partition_index+1)
            # merge current partition with previous partition
            elif partition_index in merge_prev_candidates:
                merge_pair = (partition_index-1, partition_index)           #即在prev 也在next 怎么办，直接merge to pre???????????????????????

            # reset both partitions to a minimal state
            self.network.partitions[merge_pair[0]].reset()
            self.network.partitions[merge_pair[1]].reset()

            # merge partitions
            self.network.merge(merge_pair)

            # optimise the new partition
            status = self.optimise_single_partition_Huffman(merge_pair[0],Huffmanencode_rate, Huffmandecode_rate)

            # only keep if it can merge, and the performance is better
            if not status or self.network.eval_cost() >= cost:
                self.network.partitions = partitions
                reject_list.append(merge_pair)
                logging.info(f"merging {merge_pair[0]} with {merge_pair[1]} rejected")
                print("merging not accepted")
            else:
                for i, merge in enumerate(reject_list):
                    if merge[0] >= merge_pair[1]:
                        reject_list[i] = (merge[0]-1,merge[1]-1)
                logging.info(f"merging {merge_pair[0]} with {merge_pair[1]} accepted")
                print("merging accepted")

            
            # increment while loop iterration index 
            i+=1

    def optimise(self):

        # optimise the single partitions on their own
        for partition_index in tqdm(range(len(self.network.partitions)), desc="optimising  partitions"):
            print(f"Partition {partition_index}:\n------------\n")
            self.optimise_single_partition(partition_index)
            self.network.partitions[partition_index].showing_coarse()

        # merge partitions
        print("Optimizing each partition done, start merging......")
        self.merge_partitions()

    def optimise_multi_FPGA(self):
        # optimise the single partitions on their own
        for partition_index in tqdm(range(len(self.network.partitions)), desc="optimising  partitions"):
            print(f"Partition {partition_index}:\n------------\n")

            self.optimise_single_partition(partition_index)#中间包括一步检查resource, BW.

        # merge partitions
        print("Optimizing each partition on multi FPGA done, start merging......")
        self.merge_partitions()
        
    def optimise_multi_FPGA_Huffman(self, Huffmanencode_rate, Huffmandecode_rate):
        # optimise the single partitions on their own
        for partition_index in tqdm(range(len(self.network.partitions)), desc="optimising  partitions"):
            print(f"Partition {partition_index}:\n------------\n")

            self.optimise_single_partition_Huffman(partition_index, Huffmanencode_rate, Huffmandecode_rate)#中间包括一步检查resource, BW.

        # merge partitions
        print("Optimizing each partition on multi FPGA done, start merging......")
        self.merge_partitions_Huffman(Huffmanencode_rate, Huffmandecode_rate)
        
        