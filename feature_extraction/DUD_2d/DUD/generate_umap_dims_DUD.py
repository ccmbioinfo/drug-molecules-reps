import umap
import numpy as np
import pandas as pd
# import ast
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import pad


#For DUD dataset
target_list = {
            #     1: 'ace', 
            #    2: 'ache',
            #    3: 'ar',
            #    4: 'cdk2',
            #    5: 'cox2', 
               6: 'dhfr'
            #    7: 'egfr',
            #    8: 'er_agonist',
            #    9: 'fgfr1',
            #    10: 'fxa',
            #    11: 'gpb',
            #    12: 'gr',
            #    13: 'hivrt',
            #    14: 'inha',
            #    15: 'na',
            #    16: 'p38',
            #    17: 'parp',
            #    18: 'pdgfrb',
            #    19: 'sahh',
            #    20: 'src',
            #    21:'vegfr2'
               }
# input_string = target_list[6]
input_path = "feature_outputs/"
output_path = "umap_outputs/"
dataset = 'datafiles/'

model1 = "msb-roshan/molgpt"
model2 = "gayane/BARTSmiles"#ERRORs out
model3 = "zjunlp/MolGen-7b" #Needs SELFIES notation, ERRORs out
model4 = "DeepChem/ChemBERTa-77M-MTR" 
model5 = "ncfrey/ChemGPT-1.2B"

def load_data(target, model):
    filepath = input_path + "features_"+ target + "_" + model + ".pt"
    print(f'Filename: {filepath}')
    features = torch.load(filepath)
    # print(features[0].shape)
    # print(features[1].shape)
    # print(features[2].shape)
    # dimensions = [f.ndim for f in features]
    # print(f"Dimensions: {set(dimensions)}") #unique numbers
    print(len(features))
    return features

def preprocess_tensors(tensor_list):
    flattened_list = [tensor.view(tensor.size(0), -1) for tensor in tensor_list]
    max_features = max(tensor.size(1) for tensor in flattened_list) #TODO: assuming that dim 1 is the dimension which is different
    # Pad tensors to have the same number of features
    padded_tensors = [pad(tensor, (0, max_features - tensor.size(1))) for tensor in flattened_list]
    concatenated_tensors = torch.cat(padded_tensors, dim=0)
    return concatenated_tensors

def get_labels(target):
    #separator for columns in these files is tab
    smiles_actives = pd.read_csv(dataset + "cmp_list_DUD_" + target + "_actives.dat", sep='\t')
    smiles_decoys = pd.read_csv(dataset + "cmp_list_DUD_" + target + "_decoys.dat", sep='\t')
    # print(f'Len of actives: {len(smiles_actives)}')
    # print(f'Len of decoys: {len(smiles_decoys)}')
    smiles_actives['Labels'] = 'actives'
    smiles_decoys['Labels'] = 'decoys'
    new_df = pd.concat([smiles_actives, smiles_decoys], axis=0, ignore_index=True)
    # print(f'New Labels: {len(new_df)} {new_df}')
    return new_df['Labels']

def umap_reducer(list, target, model):
    tensors = preprocess_tensors(list)
    data = tensors.numpy()
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data) #reduces to 2D data
    df = pd.DataFrame(embedding, columns=['Dim1', 'Dim2'])
    # print(f"DF: {df}")
    df['Labels'] = get_labels(target)
    print(f"df: {df}")
    df.to_csv(output_path + "embeddings_" + target + "_" + model + ".csv") #No labels in this dataset, CAS numbers are there, that too missing for some rows



# model = model1
# model = model4
model = model5
outfile = model.split('/')[1]
tensor_list = load_data(target_list[6], outfile)
umap_reducer(tensor_list, target_list[6], outfile)

#for reducing dimensions for all files
# for target in target_list.values():
#     for m in [model1, model4, model5]:
#         model = m.split('/')[1]
#         print(f'Running: {target} with {model}')
#         tensor_list = load_data(target, model)
#         umap_reducer(tensor_list, target, model)

