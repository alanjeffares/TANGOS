import arff
import os
import re
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets


DEVICE = 'cuda:0'


def get_path(device):
    if re.match("cuda:[0-9]", device):
        return '/mnt/data/shared/latent-sniper/'
    elif device == 'cpu':
        return os.path.dirname(__file__) + '/data/'
    else:
        raise ValueError(f'Device {device} not matched to known devices')


def load_boston(seed=0, train_prop=0.8, batch_size=64):
    boston = datasets.load_boston()
    X, y = boston.data, boston.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop, random_state=seed)

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1,1)).reshape(-1)
    y_test = y_scaler.transform(y_test.reshape(-1,1)).reshape(-1)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders

def load_wine(seed, train_prop=0.8, batch_size=64):
    data = pd.read_csv(get_path(DEVICE) + 'winequality-red.csv')
    data = data[:1000]

    X = data.drop('quality', axis=1)
    y = data.quality
    X, y = X.to_numpy(), y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop, random_state=seed)

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1,1)).reshape(-1)
    y_test = y_scaler.transform(y_test.reshape(-1,1)).reshape(-1)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders

def load_facebook(seed, train_prop=0.8, batch_size=64):
    data = pd.read_csv(get_path(DEVICE) + 'dataset_facebook.csv', sep=';')
    data.dropna(inplace=True)  # drop missing values
    one_hot = pd.get_dummies(data['Type']) # onehotencode categorical column
    data = data.drop('Type', axis=1)
    data = data.join(one_hot)
    X = data.drop('Lifetime Post Total Impressions', axis = 1)
    y = data['Lifetime Post Total Impressions']
    X, y = X.to_numpy(), y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop, random_state=seed)

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1,1)).reshape(-1)
    y_test = y_scaler.transform(y_test.reshape(-1,1)).reshape(-1)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders


def load_bioconcentration(seed, train_prop=0.8, batch_size=64):
    data = pd.read_csv(get_path(DEVICE) + 'Grisoni_et_al_2016_EnvInt88.csv', sep=',')

    X = data[['nHM', 'piPC09', 'PCD', 'X2Av', 'MLOGP', 'ON1V', 'N-072', 'B02[C-N]', 'F04[C-O]']]
    for var in ['nHM', 'N-072', 'B02[C-N]', 'F04[C-O]']:
        one_hot = pd.get_dummies(X[var], prefix=var) # onehotencode categorical column
        X = X.drop(var, axis=1)
        X = X.join(one_hot)

    y = data['logBCF']
    X, y = X.to_numpy(), y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop, random_state=seed)

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1,1)).reshape(-1)
    y_test = y_scaler.transform(y_test.reshape(-1,1)).reshape(-1)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders

def load_student(seed, train_prop=0.8, batch_size=64):
    data = pd.read_csv(get_path(DEVICE) + 'student-por.csv', sep=';')

    X = data.drop(['G1', 'G2', 'G3'], axis = 1)
    for var in ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
                'reason', 'guardian','schoolsup', 'famsup', 'paid', 'activities',
                'nursery', 'higher', 'internet', 'romantic',]:
        one_hot = pd.get_dummies(X[var], prefix=var) # onehotencode categorical column
        X = X.drop(var, axis=1)
        X = X.join(one_hot)

    y = data['G3']
    X, y = X.to_numpy(), y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop, random_state=seed)

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1,1)).reshape(-1)
    y_test = y_scaler.transform(y_test.reshape(-1,1)).reshape(-1)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders


def load_abalone(seed, train_prop=0.8, batch_size=64):
    data = pd.read_csv(get_path(DEVICE) + 'abalone.data', sep=',', header=None)
    data = data[:1000]
    one_hot = pd.get_dummies(data[0], drop_first=True) # onehotencode categorical column
    data = data.drop(0, axis=1)
    data = data.join(one_hot)

    X = data.drop(8, axis=1)
    y = data[8]

    X, y = X.to_numpy(), y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop, random_state=seed)

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1,1)).reshape(-1)
    y_test = y_scaler.transform(y_test.reshape(-1,1)).reshape(-1)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders


def load_skillcraft(seed, train_prop=0.8, batch_size=64):
    data = pd.read_csv(get_path(DEVICE) + 'SkillCraft1_Dataset.csv', sep=',')
    data = data.replace('?', np.NaN)
    data = data.dropna()
    data = data[:1000]
    X = data.drop(['GameID', 'LeagueIndex'], axis=1)
    y = data['LeagueIndex']

    X, y = X.to_numpy(), y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop, random_state=seed)

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders


def load_weather(seed, train_prop=0.8, batch_size=64):
    data = pd.read_csv(get_path(DEVICE) + 'Bias_correction_ucl.csv', sep=',')
    data = data.dropna()
    data = data[:1000]
    one_hot = pd.get_dummies(data['station'], drop_first=True)  # onehotencode categorical column
    data = data.drop('station', axis=1)
    data = data.join(one_hot)
    X = data.drop(['Date', 'Next_Tmax', 'Next_Tmin'], axis=1)
    y = data['Next_Tmax']

    X, y = X.to_numpy(), y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop, random_state=seed)

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders


def load_forest(seed, train_prop=0.8, batch_size=64):
    data = pd.read_csv(get_path(DEVICE) + 'forestfires.csv', sep=',')
    X = data.drop('area', axis=1)
    for var in ['X', 'Y', 'month', 'day']:
        one_hot = pd.get_dummies(X[var], prefix=var, drop_first=True)  # onehotencode categorical column
        X = X.drop(var, axis=1)
        X = X.join(one_hot)
    y = np.log(data['area'] + 1)

    X, y = X.to_numpy(), y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop, random_state=seed)

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders

def load_protein(seed, train_prop=0.8, batch_size=64):
    data = pd.read_csv(get_path(DEVICE) + 'CASP.csv', sep=',')
    data = data[:1000]
    X = data.drop('RMSD', axis=1)
    y = data['RMSD']

    X, y = X.to_numpy(), y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop, random_state=seed)

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1,1)).reshape(-1)
    y_test = y_scaler.transform(y_test.reshape(-1,1)).reshape(-1)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders


def load_heart(seed, train_prop=0.8, batch_size=64):
    data = pd.read_csv(get_path(DEVICE) + 'heart.dat', sep=' ', header=None)
    X = data.drop(13, axis=1)
    y = data[13] - 1
    data[9] = np.log(data[9] + 1)
    for var in [1, 2, 5, 6, 8, 10, 11, 12]:
        one_hot = pd.get_dummies(X[var], prefix=var, drop_first=True)  # onehotencode categorical column
        X = X.drop(var, axis=1)
        X = X.join(one_hot)

    X, y = X.to_numpy(), y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop, random_state=seed)

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders


def load_breast(seed, train_prop=0.8, batch_size=64):
    data = pd.read_csv(get_path(DEVICE) + 'breast-cancer-wisconsin.data', header=None)
    X = data.drop([0, 10], axis=1)
    X[6].replace('?', np.nan, inplace=True)
    X[6].fillna((X[6].median()), inplace=True)
    y = data[10]/2 - 1

    X, y = X.to_numpy(), y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop, random_state=seed)

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders

def load_cervical(seed, train_prop=0.8, batch_size=64):
    data = pd.read_csv(get_path(DEVICE) + 'risk_factors_cervical_cancer.csv')
    X = data[['Age', 'Number of sexual partners', 'First sexual intercourse',
          'Num of pregnancies', 'Smokes', 'Smokes (years)', 'Smokes (packs/year)',
          'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD',
          'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis',
          'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
          'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
          'STDs:pelvic inflammatory disease', 'STDs:genital herpes',
          'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV',
          'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis',
          'STDs: Time since first diagnosis', 'STDs: Time since last diagnosis',
          'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx']]
    X = X.replace('?', np.nan)
    X.fillna((X.median()), inplace=True)
    mapping = {'Hinselmann': 1, 'Schiller': 2, 'Citology': 3, 'Biopsy': 4}
    y = data[['Hinselmann', 'Schiller', 'Citology', 'Biopsy']].idxmax(axis=1)
    y = y.replace(mapping)

    for var in ['Smokes', 'Smokes (years)', 'Smokes (packs/year)', 'Hormonal Contraceptives',
                'IUD', 'STDs', 'STDs:condylomatosis', 'STDs:cervical condylomatosis',
                'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis',
                'STDs:syphilis','STDs:pelvic inflammatory disease', 'STDs:genital herpes',
                'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV', 'STDs:Hepatitis B',
                'STDs:HPV', 'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx']:
        one_hot = pd.get_dummies(X[var], prefix=var, drop_first=True)  # onehotencode categorical column
        X = X.drop(var, axis=1)
        X = X.join(one_hot)

    X, y = X.to_numpy(), y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop, random_state=seed)

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders

def load_credit(seed, train_prop=0.8, batch_size=64):
    data = pd.read_csv(get_path(DEVICE) + "crx.data", header = None)
    data = data[data[13] != '?']
    data[13] = np.log(data[13].astype(int) + 1)
    data[14] = np.log(data[14] + 1)
    data[7] = np.log(data[7] + 1)
    data[1].replace('?', np.nan, inplace=True)
    data[1].fillna((data[1].median()), inplace=True)
    X = data.drop(15, axis=1)
    y = data[15]
    mapping = {'+': 1, '-': 0}
    y.replace(mapping, inplace=True)
    for var in [0, 3, 4, 5, 6, 8, 9, 11, 12]:
        one_hot = pd.get_dummies(X[var], prefix=var, drop_first=True)  # onehotencode categorical column
        X = X.drop(var, axis=1)
        X = X.join(one_hot)

    X, y = X.to_numpy(), y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop, random_state=seed)

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders


def load_hcv(seed, train_prop=0.8, batch_size=64):
    data = pd.read_csv(get_path(DEVICE) + "hcvdat0.csv", index_col=0)
    y = data['Category'].apply(lambda x: int(x[0]))
    X = data.drop('Category', axis=1)
    one_hot = pd.get_dummies(X['Sex'], prefix='Sex', drop_first=True)  # onehotencode categorical column
    X = X.drop('Sex', axis=1)
    X = X.join(one_hot)
    X = X.fillna(X.mean())

    X, y = X.to_numpy(), y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop, random_state=seed)

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders


def load_tumor(seed, train_prop=0.8, batch_size=64):
    data = pd.read_csv(get_path(DEVICE) + 'primary-tumor.data', header=None)
    y = data[0] - 1
    X = data.drop(0, axis=1)
    for var in X.columns:
        one_hot = pd.get_dummies(X[var], prefix=var, drop_first=True)  # onehotencode categorical column
        X = X.drop(var, axis=1)
        X = X.join(one_hot)

    X, y = X.to_numpy(), y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop, random_state=seed)

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders

def load_soybean(seed, train_prop=0.8, batch_size=64):
    data0 = pd.read_csv(get_path(DEVICE) + 'soybean-large.data', header=None)
    data1 = pd.read_csv(get_path(DEVICE) + 'soybean-large.test', header=None)
    data = pd.concat([data0, data1], axis=0)
    data.reset_index(inplace=True)
    y = data[0].rank(method='dense', ascending=False).astype(int) - 1
    X = data.drop(0, axis=1)
    for var in X.columns:
        one_hot = pd.get_dummies(X[var], prefix=f'dum_{var}', drop_first=True)  # onehotencode categorical column
        X = X.drop(var, axis=1)
        X = X.join(one_hot)

    X, y = X.to_numpy(), y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop, random_state=seed)

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders

def load_australian(seed, train_prop=0.8, batch_size=64):
    data = pd.read_csv(get_path(DEVICE) + 'australian.dat', sep=' ', header=None)
    data[2] = np.log(data[2] + 1)
    data[6] = np.log(data[6] + 1)
    data[12] = np.log(data[12] + 1)
    data[13] = np.log(data[13] + 1)
    y = data[14]
    X = data.drop(14, axis=1)
    for var in [0,3,4,5,7,8,9,10,11]:
        one_hot = pd.get_dummies(X[var], prefix=var, drop_first=True)  # onehotencode categorical column
        X = X.drop(var, axis=1)
        X = X.join(one_hot)
    X, y = X.to_numpy(), y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop, random_state=seed)

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders

def load_entrance(seed, train_prop=0.8, batch_size=64):
    data_arff = arff.load(open(get_path(DEVICE) + 'CEE_DATA.arff'))
    data = pd.DataFrame(data_arff['data'])
    y = data[0].rank(method='dense', ascending=False).astype(int) - 1
    X = data.drop(0, axis=1)
    for var in X.columns:
        one_hot = pd.get_dummies(X[var], prefix=var, drop_first=True)  # onehotencode categorical column
        X = X.drop(var, axis=1)
        X = X.join(one_hot)
    X, y = X.to_numpy(), y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop, random_state=seed)

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders

def load_thoracic(seed, train_prop=0.8, batch_size=64):
    data_arff = arff.load(open(get_path(DEVICE) + 'ThoraricSurgery.arff'))
    data = pd.DataFrame(data_arff['data'])
    y = data[16]
    y = y.replace({'T':1, 'F':0})
    X = data.drop(16, axis=1)
    X[2] = np.log(X[2] + 1)
    for var in [0,3,4,5,6,7,8,9,10,11,12,13,14]:
        one_hot = pd.get_dummies(X[var], prefix=var, drop_first=True)  # onehotencode categorical column
        X = X.drop(var, axis=1)
        X = X.join(one_hot)
    X, y = X.to_numpy(), y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop, random_state=seed)

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    loaders = {
        'train': train_dataset,

        'test': DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
    }
    return loaders