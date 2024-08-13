import time
import torch
import pandas as pd

parent_folder = "DUD_2d/DUD/feature_outputs/"
input_filename = "features_labels_ace_MolGen-7b.pt"

def read_db_file():
    embeddings = torch.load(parent_folder + input_filename)
    print(f"Embeddings: {embeddings['tensors'][1].shape} \n{embeddings['tensors'][0]} ")
    target = embeddings['tensors'][0] #dummy
    return embeddings, target

def find_euclidean():
    st = time.time()
    dummy = torch.tensor([[2.,3.,4.], [4.,5.,6.], [1.,2.,3.]])
    target = torch.tensor([[2.,3.5,4.], [1.,2.5,3.]])
    dist = torch.cdist(dummy, target, p=2)
    et = time.time()
    print(f"Time: {et-st}")
    minDist = torch.argmin(dist)
    print(f"Smallest distance at: {minDist} \nfor {dist}")

def find_cosine_similarity():
    st = time.time()
    dummy = torch.tensor([[2.,3.,4.], [4.,5.,6.], [1.,2.,3.]])
    target = torch.tensor([2.,3.,4.])
    dist = torch.nn.functional.cosine_similarity(dummy, target)
    et = time.time()
    print(f"Time: {et-st}")
    minDist = torch.argmax(dist)
    print(f"Max similarity at: {minDist} \nfor {dist}")

# def find_pairwise():

read_db_file()

# find_euclidean()
# find_cosine_similarity()
