from transformers import pipeline
import pandas as pd
import torch
import numpy as np
import time
import selfies as sf


hpf_path = "/hpf/largeprojects/ccmbio/monikas/packages/"

#For Ames dataset
# parent_folder = "Ames_smi/"
# input_file = parent_folder + "Ames.csv"
# smile_column = "Canonical_Smiles"

#For BBBP dataset
# parent_folder = "bbbp_smi/"
# input_file = parent_folder + "BBBP.csv"
# is_BBBP = True
# smile_column = "smiles"

#For ClinTox dataset
parent_folder = "ClinTox_smi/"
input_file = parent_folder + "ClinTox.csv"
smile_column = "smiles"

output_path = parent_folder + "feature_outputs/"

model1 = "msb-roshan/molgpt"
model2 = "gayane/BARTSmiles"
model3 = "zjunlp/MolGen-7b" #Needs SELFIES notation
model4 = "DeepChem/ChemBERTa-77M-MTR"
model5 = "ncfrey/ChemGPT-1.2B"


def get_data(is_BBBP=False):
    if is_BBBP:
        encoding='latin-1'
    else:
        encoding = 'utf-8'
    df = pd.read_csv(input_file, encoding=encoding)
    return df

def get_features(extractor, conversion_to_selfie):
    df = get_data()
    if conversion_to_selfie:
        df[smile_column] = df[smile_column].apply(sf.encoder)
    features = extractor(df[smile_column].tolist(), return_tensors = "pt")
    return features

def feature_extraction(model_path, conversion_to_selfie=False):
    print("Running Model:", model_path)
    outfile = model_path.split('/')[1]
    extractor = pipeline("feature-extraction", framework="pt", model = model_path, model_kwargs={'cache_dir': hpf_path})
    features = get_features(extractor, conversion_to_selfie)
    out_filename = output_path + "features_"+ outfile + ".pt"
    torch.save(features, out_filename)


start_time = time.time()

feature_extraction(model1)
# feature_extraction(model2) #ERROR: unexpected 'token_type_ids'
# features = feature_extraction(model3, True)  #Errors out in mapping model file in pipeline statement I think
# feature_extraction(model4) #ChemBERT
# feature_extraction(model5) #ChemGPT

end_time = time.time()
total_time = end_time-start_time
print("Time:", total_time)

