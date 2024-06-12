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
model3 = "zjunlp/MolGen-7b" #Needs SELFIES notation
model4 = "DeepChem/ChemBERTa-77M-MTR" 
model5 = "ncfrey/ChemGPT-1.2B"

def load_data(outfile):
    filepath = input_path + "features_labels_"+ outfile + ".pt"
    data = torch.load(filepath)
    features = data['tensors']
    labels = data['labels']
    print(f"Pt file read: \n{len(features)}")
    print(f"Labels: \n{len(labels)}")
    return features, labels

def preprocess_tensors(tensor_list):
    flattened_list = [tensor.view(tensor.size(0), -1) for tensor in tensor_list]
    max_features = max(tensor.size(1) for tensor in flattened_list) #TODO: assuming that dim 1 is the dimension which is different
    # Pad tensors to have the same number of features
    padded_tensors = [pad(tensor, (0, max_features - tensor.size(1))) for tensor in flattened_list]
    concatenated_tensors = torch.cat(padded_tensors, dim=0)
    return concatenated_tensors

def umap_reducer(list, labels, outfile):
    tensors = preprocess_tensors(list)
    data = tensors.numpy()
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data) #reduces to 2D data
    df = pd.DataFrame(embedding, columns=['Dim1', 'Dim2'])
    df['Labels'] = labels
    print(f"df: {df}")
    df.to_csv(output_path + "embeddings_" + outfile + ".csv")

def gen_umap(model):
    outfile = model.split('/')[1]
    tensor_list, labels = load_data(outfile)
    umap_reducer(tensor_list, labels, outfile)

gen_umap(model1)
gen_umap(model2)
gen_umap(model3)
gen_umap(model4)
gen_umap(model5)

