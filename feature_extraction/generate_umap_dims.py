import umap
import numpy as np
import pandas as pd
# import ast
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import pad


#For Ames dataset
# parent_folder = "Ames_smi/"
# dataset_file = parent_folder + "Ames.csv"
# label_column = "Activity"

#For BBBP dataset
parent_folder = "bbbp_smi/"
dataset_file = parent_folder + "BBBP.csv"
is_BBBP = True
label_column = "p_np"


#For ClinTox dataset
# parent_folder = "ClinTox_smi/"
# dataset_file = parent_folder + "ClinTox.csv"
# label_column = "CT_TOX"


input_path = parent_folder + "feature_outputs/"
output_path = parent_folder + "umap_outputs/"

model1 = "msb-roshan/molgpt"
model2 = "gayane/BARTSmiles"#ERRORs out
model3 = "zjunlp/MolGen-7b" #Needs SELFIES notation, ERRORs out
model4 = "DeepChem/ChemBERTa-77M-MTR" 
model5 = "ncfrey/ChemGPT-1.2B"

def load_data(outfile):
    filepath = input_path + "features_"+ outfile + ".pt"
    features = torch.load(filepath)
    print(len(features))
    return features

def preprocess_tensors(tensor_list):
    flattened_list = [tensor.view(tensor.size(0), -1) for tensor in tensor_list]
    max_features = max(tensor.size(1) for tensor in flattened_list) #TODO: assuming that dim 1 is the dimension which is different
    # Pad tensors to have the same number of features
    padded_tensors = [pad(tensor, (0, max_features - tensor.size(1))) for tensor in flattened_list]
    concatenated_tensors = torch.cat(padded_tensors, dim=0)
    return concatenated_tensors

def get_labels(is_BBBP):
    if is_BBBP:
        encoding='latin-1'
    else:
        encoding = 'utf-8' #default
    labels = pd.read_csv(dataset_file, encoding = encoding)[label_column]
    return labels

def umap_reducer(list, is_BBBP = False):
    tensors = preprocess_tensors(list)
    data = tensors.numpy()
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data) #reduces to 2D data
    df = pd.DataFrame(embedding, columns=['Dim1', 'Dim2'])
    df['Labels'] = get_labels(is_BBBP)
    print(f"df: {df}")
    df.to_csv(output_path + "embeddings_" + outfile + ".csv") #No labels in this dataset, CAS numbers are there, that too missing for some rows



# model = model1
# model = model4
model = model5
outfile = model.split('/')[1]
tensor_list = load_data(outfile)
umap_reducer(tensor_list, is_BBBP)
