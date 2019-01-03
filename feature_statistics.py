import pandas as pd
import numpy as np
dFrame = pd.read_csv('datasets/facies_vectors.csv',
                         dtype={'Facies': int, 'Formation': str, 'Well Name': str, 'Depth': float, 'GR': float,
                                'ILD_log10': float, 'DeltaPHI': float, 'PHIND': float, 'PE': float, 'NM_M': int,
                                'RELPOS': float})
features = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']

dFrame = dFrame[features]

missing_valuse_dict ={
    'GR':0,
     'ILD_log10':0,
     'DeltaPHI':0,
     'PHIND':0,
     'PE':0,
     'NM_M':0,
     'RELPOS':0
}

print ("Missing values per feature")
for key in missing_valuse_dict:
    for record in dFrame[key]:
        if np.isnan(record):
            missing_valuse_dict[key] += 1
    print(key, missing_valuse_dict[key])

features_means_dict ={
    'GR':0,
     'ILD_log10':0,
     'DeltaPHI':0,
     'PHIND':0,
     'PE':0,
     'NM_M':0,
     'RELPOS':0
    }

print("*************************************************************************************************************")
print ("features Means")
for key in features_means_dict:
    features_means_dict[key] = np.mean(dFrame[key])
    print(key+" mean: "+str(features_means_dict[key]))

features_max_dict ={
    'GR':0,
     'ILD_log10':0,
     'DeltaPHI':0,
     'PHIND':0,
     'PE':0,
     'NM_M':0,
     'RELPOS':0
}

print("*************************************************************************************************************")
print ("features Maxs")
for key in features_max_dict:
    features_max_dict[key] = np.max(dFrame[key])
    print(key+" max: "+str(features_max_dict[key]))

features_min_dict ={
    'GR':0,
     'ILD_log10':0,
     'DeltaPHI':0,
     'PHIND':0,
     'PE':0,
     'NM_M':0,
     'RELPOS':0
}
print("*************************************************************************************************************")
print ("features Mins")
for key in features_min_dict:
    features_min_dict[key] = np.min(dFrame[key])
    print(key+" min: "+str(features_min_dict[key]))

features_sd_dict ={
        'GR':0,
     'ILD_log10':0,
     'DeltaPHI':0,
     'PHIND':0,
     'PE':0,
     'NM_M':0,
     'RELPOS':0
}
print("*************************************************************************************************************")
print ("features SDs")
for key in features_sd_dict:
    features_sd_dict[key] = np.std([i for i in dFrame[key] if not np.isnan(i)])
    print(key+" standard deviation: "+str(features_sd_dict[key]))

features_median_dict ={
     'GR':0,
     'ILD_log10':0,
     'DeltaPHI':0,
     'PHIND':0,
     'PE':0,
     'NM_M':0,
     'RELPOS':0
}
print("*************************************************************************************************************")
print ("features Medians")
for key in features_median_dict:
    features_median_dict[key] = np.median([i for i in dFrame[key] if not np.isnan(i)])
    print(key+" median: "+str(features_median_dict[key]))