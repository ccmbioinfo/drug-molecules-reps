import umap
import numpy as np
import pandas as pd
# import ast
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import pad
from transformers import AutoTokenizer

is_BARTSmiles = False
is_BBBP = False

#For Ames dataset
# parent_folder = "Ames_smi/"
# dataset_file = parent_folder + "Ames.csv"
# label_column = "Activity"
# smile_column = "Canonical_Smiles"


#For BBBP dataset
parent_folder = "bbbp_smi/"
dataset_file = parent_folder + "BBBP.csv"
is_BBBP = True
label_column = "p_np"
smile_column = "smiles"


#For ClinTox dataset
# parent_folder = "ClinTox_smi/"
# dataset_file = parent_folder + "ClinTox.csv"
# label_column = "CT_TOX"
# smile_column = "smiles"


input_path = parent_folder + "feature_outputs/"
output_path = parent_folder + "umap_outputs/"

model1 = "msb-roshan/molgpt"
model2 = "gayane/BARTSmiles"
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

def remove_long_molecules(df):
    """to handle the index out of range error due to max positional embedding limit of 128 in Model2, long molecules with token length > 128 are removed"""
    tokenizer = AutoTokenizer.from_pretrained(model2)
    mask = df[smile_column].apply(lambda smiles: len(tokenizer(smiles).input_ids) > 128)
    counter = mask.sum()
    filtered = df[~mask].reset_index(drop=True)
    print(f"Counter: {counter}, {len(df)}, \nfraction: {(counter/len(df))*100}" )
    return filtered

def get_labels():
    if is_BBBP:
        encoding='latin-1'
    else:
        encoding = 'utf-8' #default
    df = pd.read_csv(dataset_file, encoding = encoding)
    if is_BARTSmiles: #to filter the long molecules
        df = remove_long_molecules(df)
    print(f"Labels: {len(df)} {df}")
    return df[label_column]

def umap_reducer(list):
    labels = get_labels()
    tensors = preprocess_tensors(list)
    data = tensors.numpy()
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data) #reduces to 2D data
    df = pd.DataFrame(embedding, columns=['Dim1', 'Dim2'])
    df['Labels'] = labels
    print(f"df: {df}")
    df.to_csv(output_path + "embeddings_" + outfile + ".csv")


# model = model1
model = model2
is_BARTSmiles = True
# model = model4
# model = model5
outfile = model.split('/')[1]
tensor_list = load_data(outfile)
umap_reducer(tensor_list)
