import argparse
import subprocess
import sys
import os
from llamafactory.train.tuner import run_exp
import torch
import pandas as pd
import json

# parser = argparse.ArgumentParser(
#                     prog='ProgramName',
#                     description='What the program does',
#                     epilog='Text at the bottom of help')

# parser.add_argument('-d', '--dataset_path')
# parser.add_argument('-c', '--config_file')

# args = parser.parse_args()
# print(args.dataset_path)


dataset_path = '/cos_mount/users/dibyanayan/data_for_ept_python_sample.json'


# df = []


# for parquet_files in os.listdir(dataset_path):
#     parquet = pd.read_parquet(os.path.join(dataset_path,parquet_files))
#     for index, row in parquet.iterrows():
#         repo = row['content']
#         lines = repo.strip().split('\n')
#         i = 0
        
#         while i < len(lines):
#             if lines[i].startswith("<filename>"):
#                 filepath = lines[i].replace("<filename>", "").strip()
#                 file_content = []
#                 i += 1
#                 while i < len(lines) and not lines[i].startswith("<filename>"):
#                     file_content.append(lines[i])
#                     i += 1
#                 file_content = '\n'.join(file_content)
            
#         df.append({'text': file_content})


with open(dataset_path, 'rb') as handle:
    df = json.load(handle)

    
    
print(len(df))
print(df[:4])
with open("./data/c4_demo.json", "w") as final:
    json.dump(df, final)
     



    





def get_device_count() -> int:
    r"""
    Gets the number of available GPU or NPU devices.
    """
    
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        return 0




def launch():
    run_exp()


if __name__ == "__main__":
    launch()



# print(launcher.__file__)
# # # Define the command to be executed
# # command = "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train train_llama3.json"

# # # Execute the command
# # subprocess.run(command, shell=True, check=True)

# force_torchrun = os.environ.get("FORCE_TORCHRUN", "0").lower() in ["true", "1"]
# if force_torchrun or get_device_count() > 1:
#     master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
#     master_port = os.environ.get("MASTER_PORT", str(random.randint(20001, 29999)))
#     logger.info("Initializing distributed tasks at: {}:{}".format(master_addr, master_port))
#     process = subprocess.run(
#         (
#             "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
#             "--master_addr {master_addr} --master_port {master_port} {file_name} {args}"
#         ).format(
#             nnodes=os.environ.get("NNODES", "1"),
#             node_rank=os.environ.get("RANK", "0"),
#             nproc_per_node=os.environ.get("NPROC_PER_NODE", str(get_device_count())),
#             master_addr=master_addr,
#             master_port=master_port,
#             file_name=launcher.__file__,
#             args='./train_llama3.json',
#         ),
#         shell=True,
#     )
#     sys.exit(process.returncode)
