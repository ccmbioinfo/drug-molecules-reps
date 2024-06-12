from transformers import pipeline, AutoModel, AutoTokenizer
import pandas as pd
import torch
import numpy as np
import time
import selfies as sf
import os


hpf_path = "/hpf/largeprojects/ccmbio/monikas/packages/"
is_BBBP = False

#For Ames dataset
parent_folder = "Ames_smi/"
input_file = parent_folder + "Ames.csv"
smile_column = "Canonical_Smiles"
label_column = "Activity"

#For BBBP dataset
# parent_folder = "bbbp_smi/"
# input_file = parent_folder + "BBBP.csv"
# is_BBBP = True
# smile_column = "smiles"
# label_column = "p_np"

#For ClinTox dataset
# parent_folder = "ClinTox_smi/"
# input_file = parent_folder + "ClinTox.csv"
# smile_column = "smiles"
# label_column = "CT_TOX"


output_path = parent_folder + "feature_outputs/"

model1 = "msb-roshan/molgpt"
model2 = "gayane/BARTSmiles"
model3 = "zjunlp/MolGen-7b" #Needs SELFIES notation
model4 = "DeepChem/ChemBERTa-77M-MTR"
model5 = "ncfrey/ChemGPT-1.2B"


def get_data():
    if is_BBBP:
        encoding='latin-1'
    else:
        encoding = 'utf-8'
    df = pd.read_csv(input_file, encoding=encoding)
    return df

def remove_long_molecules(df):
    """SPECIFIC to BARTSmiles: to handle the index out of range error due to max positional embedding limit of 128 in Model2, long molecules with token length > 128 are removed"""
    tokenizer = AutoTokenizer.from_pretrained(model2)
    mask = df[smile_column].apply(lambda smiles: len(tokenizer(smiles).input_ids) > 128)
    counter = mask.sum()
    filtered = df[~mask].reset_index(drop=True)
    print(f"Long molecules: {counter}, {len(df[smile_column])}, \nfraction: {(counter/len(df[smile_column]))*100}" )
    return filtered

def convert_to_selfies(smile, idx, wrong_smiles, error_idx_list):
    """SPECIFIC to MolGen: To test and weed out the erroneous SMILE molecules which fail selfie conversion for Model3"""
    try:
        return sf.encoder(smile)
    except Exception as e:
        # print(f"Errored! for {smile}:\n {e}")
        wrong_smiles.append(smile)
        error_idx_list.append(idx)
        return None

def write_error_smiles_to_file(errored_smiles_list):
    output = os.path.join(parent_folder, parent_folder.split('_')[0] + "_WRONG_SMILES.txt")
    with open(output, "w") as file:
        for smile in errored_smiles_list:
            file.write(smile + "\n")

def remove_error_smiles(df):
    wrong_smiles = []
    error_idx_list = []
    df[smile_column] = df.apply(lambda row: convert_to_selfies(row[smile_column], row.name, wrong_smiles, error_idx_list), axis=1)
    if wrong_smiles: #if not empty
        print(f"Error smiles: {len(wrong_smiles)}, {len(df)}, {len(wrong_smiles)/len(df)*100}")
        write_error_smiles_to_file(wrong_smiles)
        df = df.drop(error_idx_list).reset_index(drop=True)
    return df

def get_features(extractor, is_BARTSmiles, conversion_to_selfie):
    df = get_data()
    if conversion_to_selfie: #only for model3
        # df[smile_column] = df[smile_column].apply(sf.encoder) #Original
        df = remove_error_smiles(df)
    if is_BARTSmiles: #only for model2
        df = remove_long_molecules(df)

    print(f"Running extractor...")
    features = extractor(df[smile_column].tolist(), return_tensors = "pt")
    return features, df[label_column]

def feature_extraction(model_path, is_BARTSmiles = False, conversion_to_selfie=False):
    print("Running Model:", model_path, "for:", parent_folder)
    outfile = model_path.split('/')[1]
    extractor = pipeline("feature-extraction", framework="pt", model = model_path, model_kwargs={'cache_dir': hpf_path}, tokenize_kwargs ={'return_token_type_ids': False})
    features, labels = get_features(extractor, is_BARTSmiles, conversion_to_selfie)
    # print(f"Labels after function: {labels}")
    out_filename = output_path + "features_labels_"+ outfile + ".pt"
    tensor_wth_labels = {'tensors': features, 'labels': labels}
    torch.save(tensor_wth_labels, out_filename)


start_time = time.time()

# feature_extraction(model1) #MolGPT
# feature_extraction(model2, True, False) #BARTSmiles
feature_extraction(model3, False, True) #MolGen-7b: Requires Selfies
feature_extraction(model4) #ChemBERT
feature_extraction(model5) #ChemGPT


end_time = time.time()
total_time = end_time-start_time
print("Time:", total_time)

