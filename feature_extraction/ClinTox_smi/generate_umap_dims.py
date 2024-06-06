import umap
import numpy as np
import pandas as pd
# import ast
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import pad


input_path = "./feature_outputs/"
output_path = "./umap_outputs/"

model1 = "msb-roshan/molgpt"
model2 = "gayane/BARTSmiles"#ERRORs out
model3 = "zjunlp/MolGen-7b" #Needs SELFIES notation, ERRORs out
model4 = "DeepChem/ChemBERTa-77M-MTR" 
model5 = "ncfrey/ChemGPT-1.2B"

def load_data(outfile):
    # outfile = model_path.split('/')[1]
    filepath = input_path + "features_"+ outfile + ".pt"
    features = torch.load(filepath)
    print(features[0].shape)
    print(features[1].shape)
    print(features[2].shape)
    dimensions = [f.ndim for f in features]
    print(f"Dimensions: {set(dimensions)}") #unique numbers
    print(len(features))
    # features = np.loadtxt(filepath, dtype=str)
    # print("Features:", features.shape)
    return features

    '''
    #Doesn't work
    tensor_list = []
    with open(filepath, "r") as file:
        for line in file:
            line = line.strip()
        # content = file.read()
            array = eval(line)
            array = np.fromstring(line[1:-1], sep=',') #TODO Google if this function can be used for multi dimensions array
            tensor_list.append(torch.tensor(array))
    '''

def get_labels():
    labels = pd.read_csv("ClinTox.csv").CT_TOX
    return labels

def preprocess_tensors(tensor_list):
    flattened_list = [tensor.view(tensor.size(0), -1) for tensor in tensor_list]
    max_features = max(tensor.size(1) for tensor in flattened_list) #TODO: assuming that dim 1 is the one different
    # Pad tensors to have the same number of features
    padded_tensors = [pad(tensor, (0, max_features - tensor.size(1))) for tensor in flattened_list]
    concatenated_tensors = torch.cat(padded_tensors, dim=0)
    return concatenated_tensors

def umap_reducer(list):
    # n_labels = len(list)
    tensors = preprocess_tensors(list)
    data = tensors.numpy()
    # labels = np.repeat(np.arange(n_labels), 1) #hardcoding number of samples per label as 1, since we have only 1 row per molecule in csv
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data) #reduces to 2D data
    df = pd.DataFrame(embedding, columns=['Dim1', 'Dim2'])
    df['Labels'] = get_labels()
    print(f"df: {df}")
    df.to_csv(output_path + "embeddings_" + outfile + ".csv") #No labels in this dataset, CAS numbers are there, that too missing for some rows



model = model1
model = model4
model = model5
outfile = model.split('/')[1]
tensor_list = load_data(outfile)
umap_reducer(tensor_list)
