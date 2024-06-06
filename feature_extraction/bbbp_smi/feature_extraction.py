from transformers import pipeline
import pandas as pd
import torch
import numpy as np
import time
import selfies as sf

#Dummy code for test
'''
checkpoint = "facebook/bart-base"
feature_extractor = pipeline("feature-extraction", framework="pt", model=checkpoint)
text = "Transformers is an awesome library!"

#Reducing along the first dimension to get a 768 dimensional array
features = feature_extractor(text,return_tensors = "pt")[0].numpy().mean(axis=0)

print(features)
'''
np.set_printoptions(threshold=np.inf) #for expanding ellipses in text file when saving as numpy

input_path = "BBBP.csv"
output_path = "feature_outputs/"
hpf_path = "/hpf/largeprojects/ccmbio/monikas/packages/"

def get_data():
    df = pd.read_csv(input_path, encoding='latin-1') #gave error due to an ISO-LATIN-1 code which is an invalid UTF-8 byte code
    # print("CSV:", df)
    return df

def get_features(extractor, conversion_to_selfie):
    print("Here1")
    df = get_data() #ASK if we have to concatenate columns? or just use the structure
    # print(df)
    # features = df["Canonical_Smiles"].apply(lambda x: extractor(x, return_tensors = "pt")[0].numpy().mean(axis=0))
    if conversion_to_selfie:
        df["smiles"] = df["smiles"].apply(sf.encoder) #convert to selfies for MolGen model
    # print(df["smiles"])
    features = extractor(df["smiles"].tolist(), return_tensors = "pt")
    # print(features.get_shape())
    print(features[0])
    # print(features)
    return features

def feature_extraction(model_path, conversion_to_selfie=False):
    print("\n=====================Running Model:=======================", model_path)
    outfile = model_path.split('/')[1]
    # print("Here")
    extractor = pipeline("feature-extraction", framework="pt", model = model_path, model_kwargs={'cache_dir': hpf_path})
    features = get_features(extractor, conversion_to_selfie)
    # out_filename = output_path + "features_"+ outfile + ".txt"
    out_filename = output_path + "features_"+ outfile + ".pt"
    torch.save(features, out_filename)
    '''
    with open(out_filename, "w") as file:
        for feature in features:
            # file.write(np.array2string(feature.numpy(), separator=",") + "\n")
            file.write(str(feature.numpy()) + "\n")
    '''

#For Ames dataset
model1 = "msb-roshan/molgpt"
model2 = "gayane/BARTSmiles"
model3 = "zjunlp/MolGen-7b" #Needs SELFIES notation
model4 = "DeepChem/ChemBERTa-77M-MTR"
model5 = "ncfrey/ChemGPT-1.2B"
start_time = time.time()
# feature_extraction(model1)
# feature_extraction(model2) #ERROR: BartModel.forward() got an unexpected keyword argument 'token_type_ids'
# feature_extraction(model3, True) #ERROR: Unable to load weights from pytorch checkpoint file
# feature_extraction(model4)
# feature_extraction(model5) #Takes too long, 18m!

end_time = time.time()
total_time = end_time-start_time
print("Time:", total_time)