from architectures.deepctr.models import *
from architectures.deepctr.utils import VarLenFeat, SingleFeat
from sklearn import preprocessing
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
from tensorflow.keras.utils import plot_model
from time import time
import pandas as pd
import numpy as np
import itertools
import os
import telegram_logger as tg
import logging
import warnings
import time as t

warnings.filterwarnings('ignore')

if_hash = False
target_candidate = ['label']
target_company = ['interest']
logging.basicConfig(filename=f'logs/log.txt',
                    format='%(levelname)s : %(asctime)s : %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.DEBUG,
                    filemode='a')


def log(text, if_print=False, if_log=True, if_telegram=False, logger=False):
    if if_print:
        print(text)
    if if_log:
        logger = logging.getLogger()
        logger.debug(text)
    if if_telegram:
        tg.send(text, logger=logger)


def to_list(row):
    return ''.join(str(row).split()).split('##')


def remove_undefined_gender(row):
    return row.replace(' ', '')


def clean_text(row):
    return remove_redundant(restore_commas(remove_tags(row)))


def remove_tags(row):
    return re.compile(r'<[^>]+>').sub('', row)  # EOL chars


def restore_commas(row):
    return row.replace('##', ',')


def remove_redundant(row):
    _ = re.sub(r'[\n\r\t]', ' ', row)
    return re.sub(r'[ ]+', ' ', _).strip()


def list_transform(l, le, length):
    if length == 0:
        return np.concatenate((np.array(le.transform(l)), np.zeros(length))) + 1
    elif length == 1:
        try:
            return np.concatenate((np.array(le.transform(l)), np.zeros(length)))[0] + 1
        except:
            return 0
    return np.concatenate((np.array(le.transform(l)), np.zeros(length)))[:length] + 1


def convert_to_label_list(column, length=0):
    le = preprocessing.LabelEncoder()
    if type(column.iloc[0]) == set:
        x = column.apply(lambda x: list(x))
    else:
        x = column.apply(to_list)

    le.fit(list(set(list(itertools.chain.from_iterable(x.values)))))
    return x.apply(list_transform, args=(le, length))


def get_train_id(positive_data, num_negatives=4):
    positive_data['label'] = 1
    positive_data = positive_data[['AdayId', 'IlanId', 'label', 'interest']]
    np.random.seed(19850806)
    negative_samples = pd.DataFrame(data={'AdayId': positive_data['AdayId'].repeat(num_negatives),
                                          'IlanId': np.random.choice(positive_data['IlanId'].values,
                                                                     size=positive_data.shape[0] * num_negatives),
                                          'label': 0,
                                          'interest': 0})
    data = pd.concat([positive_data, negative_samples]).drop_duplicates(subset=['AdayId', 'IlanId'], keep='first',
                                                                        inplace=False)
    data.columns = ['user', 'item', 'label', 'interest']
    return data


def shuffle_df(data):
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    return data.iloc[indices]


def extract_job_nontext_feats(jobs):
    mms = MinMaxScaler(feature_range=(0, 1))

    jobs.egitimdurumu = convert_to_label_list(jobs.egitimdurumu, 6)

    jobs.cinsiyet.fillna('', inplace=True)
    jobs.cinsiyet = jobs.cinsiyet.apply(remove_undefined_gender)
    jobs.cinsiyet = convert_to_label_list(jobs.cinsiyet, 1) - 1

    jobs.MinTecrube.fillna(0, inplace=True)  # real value
    jobs.MinTecrube[jobs.MinTecrube > 20] = 0
    jobs.MaxTecrube = jobs[['MinTecrube', 'MaxTecrube']].max(axis=1)

    jobs.iller = convert_to_label_list(jobs.iller, 10)
    jobs.sektorler = convert_to_label_list(jobs.sektorler, 2)
    jobs.pozisyonTipi = convert_to_label_list(jobs.pozisyonTipi, 1) - 1
    jobs.PozisyonSeviyesi.fillna(0, inplace=True)
    jobs.PozisyonSeviyesi = convert_to_label_list(jobs.PozisyonSeviyesi, 1)
    jobs.Lang = convert_to_label_list(jobs.Lang, 1) - 1
    jobs.Askerlik.fillna('', inplace=True)
    jobs.Askerlik = convert_to_label_list(jobs.Askerlik, 2)

    jobs.MinTecrube = mms.fit_transform(jobs.MinTecrube.values.reshape(-1, 1))
    jobs.MaxTecrube = mms.fit_transform(jobs.MaxTecrube.values.reshape(-1, 1))
    jobs.PozisyonSeviyesi = mms.fit_transform(jobs.PozisyonSeviyesi.values.reshape(-1, 1))
    jobs.IseAlinacakKisiSayisi = mms.fit_transform(jobs.IseAlinacakKisiSayisi.values.reshape(-1, 1))

    jobs.ehliyet.fillna('', inplace=True)
    jobs.ehliyet = convert_to_label_list(jobs.ehliyet, 1) - 1

    # investigate for further use
    jobs.drop(['FirmaId'], axis=1, inplace=True)
    jobs.drop(['pozisyonId'], axis=1, inplace=True)
    # text
    jobs.drop(['pozisyonAdi', 'Nitelikler', 'IlanAciklama'], axis=1, inplace=True)
    # use it later
    jobs.index = jobs['IlanId'].values
    # jobs.drop(['IlanId'], axis=1, inplace=True)
    jobs.IlanId = convert_to_label_list(jobs.IlanId, 1) - 1

    jobs.columns = ['i_id',
                    'i_education',
                    'i_gender',
                    'i_min_experience',
                    'i_max_experience',
                    'i_provinces',
                    'i_sectors',
                    'i_hidden',
                    'i_position_type',
                    'i_position_level',
                    'i_lang',
                    'i_num',
                    'i_military_service',
                    'i_driving']
    item_features = {}
    sparse_features = ['i_id', 'i_gender', 'i_hidden', 'i_position_type', 'i_lang', 'i_driving']
    item_features['sparse'] = [SingleFeat(feat, jobs[feat].nunique(), hash_flag=if_hash, dtype="float32")
                               for feat in sparse_features]

    dense_features = ['i_min_experience', 'i_max_experience', 'i_position_level', 'i_num']
    item_features['dense'] = [SingleFeat(feat, 0, ) for feat in dense_features]

    sequence_features = ['i_education', 'i_provinces', 'i_sectors', 'i_military_service']
    item_features['sequence'] = [VarLenFeat(feat,
                                            len(np.unique(np.concatenate((jobs[feat].values), axis=0))) + 30,
                                            len(jobs[feat].iloc[0]), 'mean', hash_flag=if_hash, dtype="float32")
                                 for feat in sequence_features]

    return jobs, item_features


def combine_cand_feats(candidates):
    _ = candidates.groupby('AdayId')['IstecrubesiAciklama'].apply(lambda x: ' '.join(set(x))).to_frame()
    _.rename(columns={'IstecrubesiAciklama': 'explanation'}, inplace=True)
    candidates = candidates.merge(_, on='AdayId', how='left')

    _ = candidates.groupby('AdayId')['PozisyoIsmi'].apply(lambda x: ' '.join(set(x))).to_frame()
    _.rename(columns={'PozisyoIsmi': 'position'}, inplace=True)
    candidates = candidates.merge(_, on='AdayId', how='left')

    candidates.EgitimDurumu.fillna('', inplace=True)
    _ = candidates.groupby('AdayId')['EgitimDurumu'].apply(lambda x: set(x)).to_frame()
    _.rename(columns={'EgitimDurumu': 'education'}, inplace=True)
    candidates = candidates.merge(_, on='AdayId', how='left')

    candidates.departmanKodu.fillna(9999, inplace=True)
    _ = candidates.groupby('AdayId')['departmanKodu'].apply(lambda x: set(x)).to_frame()
    _.rename(columns={'departmanKodu': 'faculty'}, inplace=True)
    candidates = candidates.merge(_, on='AdayId', how='left')

    candidates.universiteKodu.fillna(9999, inplace=True)
    _ = candidates.groupby('AdayId')['universiteKodu'].apply(lambda x: set(x)).to_frame()
    _.rename(columns={'universiteKodu': 'university'}, inplace=True)
    candidates = candidates.merge(_, on='AdayId', how='left')

    _ = candidates.groupby('AdayId')['YasadigiSehir'].apply(lambda x: set(x)).to_frame()
    _.rename(columns={'YasadigiSehir': 'city'}, inplace=True)
    candidates = candidates.merge(_, on='AdayId', how='left')

    candidates.drop(columns=['EgitimDurumu', 'IstecrubesiAciklama', 'CvNo', 'IseBaslamaTarihi',
                             'IsdenCikisTarihi', 'PozisyonId', 'PozisyoIsmi', 'departmanAdi',
                             'universiteAdi', 'departmanKodu', 'universiteKodu', 'YasadigiSehir'], inplace=True)

    candidates.drop_duplicates('AdayId', inplace=True)
    candidates.reset_index(drop=True, inplace=True)
    return candidates


def extract_cand_nontext_feats(candidates):
    mms = MinMaxScaler(feature_range=(0, 1))

    candidates.Askerlik.fillna('', inplace=True)
    candidates.Askerlik = convert_to_label_list(candidates.Askerlik, 1) - 1
    candidates.CalismaDurumu = convert_to_label_list(candidates.CalismaDurumu, 1) - 1

    candidates.Cinsiyet.fillna(-1, inplace=True)
    candidates.Cinsiyet = convert_to_label_list(candidates.Cinsiyet, 1) - 1
    candidates.Ehliyet = convert_to_label_list(candidates.Ehliyet, 1) - 1

    candidates.education = convert_to_label_list(candidates.education, 2)

    candidates.faculty = convert_to_label_list(candidates.faculty, 2)

    candidates.university = convert_to_label_list(candidates.university, 2)

    candidates.city = convert_to_label_list(candidates.city, 2)

    candidates.Yas = mms.fit_transform(candidates.Yas.values.reshape(-1, 1))

    # texts; consider them later
    candidates.drop(['explanation', 'position'], axis=1, inplace=True)

    # consider it
    candidates.index = candidates['AdayId'].copy().values
    # candidates.drop(['AdayId'], axis=1, inplace=True)
    candidates.AdayId = convert_to_label_list(candidates.AdayId, 1) - 1

    candidates.columns = ['u_id',
                          'u_military_service',
                          'u_working_state',
                          'u_gender',
                          'u_driving',
                          'u_age',
                          'u_education',
                          'u_faculty',
                          'u_university',
                          'u_city']

    user_features = {}
    sparse_features = ['u_id', 'u_military_service', 'u_working_state', 'u_gender', 'u_driving']
    user_features['sparse'] = [SingleFeat(feat, candidates[feat].nunique(), hash_flag=if_hash, dtype="float32")
                               for feat in sparse_features]

    dense_features = ['u_age']
    user_features['dense'] = [SingleFeat(feat, 0, ) for feat in dense_features]

    sequence_features = 'u_education', 'u_faculty', 'u_university', 'u_city'
    user_features['sequence'] = [VarLenFeat(feat,
                                            len(np.unique(np.concatenate((candidates[feat].values), axis=0))) + 30,
                                            len(candidates[feat].iloc[0]), 'mean', hash_flag=if_hash, dtype="float32")
                                 for feat in sequence_features]

    return candidates, user_features


def get_data():
    applications = pd.read_csv('../data/reciprocal_recommendation/applications.csv')
    interests = pd.read_csv('../data/reciprocal_recommendation/phoneviews.csv')

    interests.drop_duplicates(subset=['AdayId', 'IlanId'], keep='last', inplace=True)
    pairs = [tuple((u, i)) for (u, i) in zip(applications.AdayId, applications.IlanId)]
    interests = interests[interests.set_index(['AdayId', 'IlanId']).index.isin(pairs)]
    interests = interests[['AdayId', 'IlanId']]
    interests['interest'] = 1
    applications = applications.merge(interests, on=['AdayId', 'IlanId'], how='left').fillna(0)

    jobs = pd.read_csv('../data/reciprocal_recommendation/jobs.csv', encoding='latin5')
    items, item_features = extract_job_nontext_feats(jobs)

    candidates = pd.read_csv('../data/reciprocal_recommendation/candidates.csv', encoding='latin5')
    users, user_features = extract_cand_nontext_feats(combine_cand_feats(candidates))

    features = {}
    features['sparse'] = item_features['sparse'] + user_features['sparse']
    features['dense'] = item_features['dense'] + user_features['dense']
    features['sequence'] = item_features['sequence'] + user_features['sequence']

    data = shuffle_df(get_train_id(applications, num_negatives=2))
    data = pd.concat([items.loc[data['item']].reset_index(drop=True),
                      users.loc[data['user']].reset_index(drop=True),
                      data['label'].reset_index(drop=True),
                      data['interest'].reset_index(drop=True)], axis=1)
    train, test = train_test_split(data, test_size=0.2, random_state=19880519)
    train_model_input = [train[feat.name].values for feat in features['sparse']] + \
                        [train[feat.name].values for feat in features['dense']] + \
                        [np.array(list(map(lambda x: x.tolist(), train[feat.name].values))) for feat in
                         features['sequence']]
    test_model_input = [test[feat.name].values for feat in features['sparse']] + \
                       [test[feat.name].values for feat in features['dense']] + \
                       [np.array(list(map(lambda x: x.tolist(), test[feat.name].values))) for feat in
                        features['sequence']]

    return train, test, train_model_input, test_model_input, features


def get_callbacks(name, patience=5):
    model_file = f'models/{name}/weights.hdf5'
    checkpointer = ModelCheckpoint(filepath=model_file, save_best_only=True, save_weights_only=True)
    early_stop = EarlyStopping(patience=patience, verbose=1)
    csv_logger = CSVLogger(f'logs/{name}/training.log')
    tensorboard = TensorBoard(log_dir=f"logs/{name}/{time()}")

    return [checkpointer, early_stop, csv_logger]


def start(name):
    try:
        os.system(f"rm logs/{name}/*")
    except:
        pass
    try:
        os.system(f"rm models/{name}/*")
    except:
        pass
    try:
        os.system(f"mkdir logs/{name}")
    except:
        pass
    try:
        os.system(f"mkdir models/{name}")
    except:
        pass


def run_biobjective_model(model, name, train, test, train_model_input, test_model_input):
    start(name)
    # weight=0.2
    # weight=float(name.split('_')[1])
    start_time = t.time()
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )  # loss_weights=[1-weight, weight]
    callbacks = get_callbacks(name, 3)

    history = model.fit(train_model_input, [train[target_candidate].values, train[target_company].values],
                        batch_size=256, epochs=20, verbose=2, validation_split=0.1, callbacks=callbacks)
    model.load_weights(f"models/{name}/weights.hdf5")
    elapsed_train_time = t.strftime('%H:%M:%S', t.gmtime(t.time() - start_time))
    start_time = t.time()
    pred_ans = model.predict(test_model_input, batch_size=256)
    elapsed_test_time = t.strftime('%H:%M:%S', t.gmtime(t.time() - start_time))

    plot_model(model, show_shapes=True, show_layer_names=True, to_file=f'models/{name}/model_detailed.png')
    plot_model(model, to_file=f'models/{name}/model.png')
    log(f"{name}_candidate:: LogLoss: {round(log_loss(test[target_candidate].values, pred_ans[0]), 4)}",
        if_telegram=True, logger=True)
    log(f"{name}_candidate:: AUC: {round(roc_auc_score(test[target_candidate].values, pred_ans[0]), 4)}",
        if_telegram=True, logger=True)

    log(f"{name}_company:: LogLoss: {round(log_loss(test[target_company].values, pred_ans[1]), 4)}", if_telegram=True,
        logger=True)
    log(f"{name}_company:: AUC: {round(roc_auc_score(test[target_company].values, pred_ans[1]), 4)}", if_telegram=True,
        logger=True)
    log(f"{name}:: train-time: {elapsed_train_time}", if_telegram=True, logger=True)
    log(f"{name}:: test-time: {elapsed_test_time}", if_telegram=True, logger=True)


def run_model(model, name, train, test, train_model_input, test_model_input, target):
    target_name = 'candidate' if target == target_candidate else 'company'
    start(name)
    start_time = t.time()
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )
    callbacks = get_callbacks(name, 3)

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=20, verbose=2, validation_split=0.1, callbacks=callbacks)
    model.load_weights(f"models/{name}/weights.hdf5")
    elapsed_train_time = t.strftime('%H:%M:%S', t.gmtime(t.time() - start_time))
    start_time = t.time()
    pred_ans = model.predict(test_model_input, batch_size=256)
    elapsed_test_time = t.strftime('%H:%M:%S', t.gmtime(t.time() - start_time))

    plot_model(model, show_shapes=True, show_layer_names=True, to_file=f'models/{name}/model_detailed.png')
    plot_model(model, to_file=f'models/{name}/model.png')
    log(f"{name}_{target_name}:: LogLoss: {round(log_loss(test[target].values, pred_ans), 4)}", if_telegram=True,
        logger=True)
    log(f"{name}_{target_name}:: AUC: {round(roc_auc_score(test[target].values, pred_ans), 4)}", if_telegram=True,
        logger=True)
    log(f"{name}_{target_name}:: train-time: {elapsed_train_time}", if_telegram=True, logger=True)
    log(f"{name}_{target_name}:: test-time: {elapsed_test_time}", if_telegram=True, logger=True)


def run():
    train, test, train_model_input, test_model_input, features = get_data()
    
    # biDeepFM
    model = biDeepFM(features, task='binary')
    run_biobjective_model(model, 'biDeepFM', train, test, train_model_input, test_model_input)
    """
    # biDeepFM --to set alpha (weight of )
    for i in np.arange(0.2,0.25,0.1):
        model = biDeepFM(features, task='binary')
        run_biobjective_model(model, f'biDeepFM_{i}', train, test,  train_model_input, test_model_input)
    """
    return model


def run_AttbiDeepFM():
    train, test, train_model_input, test_model_input, features = get_data()

    # AttbiDeepFM
    model = AttbiDeepFM(features, task='binary', dnn_hidden_units=())
    run_biobjective_model(model, 'AttbiDeepFM', train, test, train_model_input, test_model_input)

    return model


def run_biFM():
    train, test, train_model_input, test_model_input, features = get_data()

    # biFM
    model = biDeepFM(features, task='binary', dnn_hidden_units=())
    run_biobjective_model(model, 'biFM', train, test, train_model_input, test_model_input)

    return model


def run_biDNN():
    train, test, train_model_input, test_model_input, features = get_data()

    # biDNN
    model = biDeepFM(features, task='binary', use_fm=False, only_dnn=True)
    run_biobjective_model(model, 'biDNN', train, test, train_model_input, test_model_input)

    return model


def run_biDeepFM_4CF():
    train, test, train_model_input, test_model_input, features = get_data()

    train = train[['i_id', 'u_id', 'label', 'interest']]
    test = test[['i_id', 'u_id', 'label', 'interest']]

    train_model_input = [train_model_input[0], train_model_input[6]]
    test_model_input = [test_model_input[0], test_model_input[6]]

    features['sparse'] = [features['sparse'][0], features['sparse'][6]]
    features['dense'] = []
    features['sequence'] = []

    # biDeepFM_4CF
    model = biDeepFM(features, task='binary')
    run_biobjective_model(model, 'biDeepFM_4CF', train, test, train_model_input, test_model_input)

    return model


def test_embedding_size():
    train, test, train_model_input, test_model_input, features = get_data()

    for i in np.arange(2, 25, 2):
        print(f'*** Testing embedding size={i} ***')
        # biDeepFM
        model = biDeepFM(features, task='binary', embedding_size=i)
        run_biobjective_model(model, f'biDeepFM_e{i}', train, test, train_model_input, test_model_input)

    return model


def run_all_single_objective_models():
    train, test, train_model_input, test_model_input, features = get_data()

    # PNN
    model = PNN(features, task='binary')
    run_model(model, 'PNN', train, test, train_model_input, test_model_input, target_candidate)
    model = PNN(features, task='binary')
    run_model(model, 'PNN_', train, test, train_model_input, test_model_input, target_company)
    
    # DeepFM
    model = DeepFM(features, task='binary') # final_activation='sigmoid', dnn_hidden_units=[]
    run_model(model, 'DeepFM', train, test,  train_model_input, test_model_input, target_candidate)
    model = DeepFM(features, task='binary') # final_activation='sigmoid', dnn_hidden_units=[]
    run_model(model, 'DeepFM_', train, test,  train_model_input, test_model_input, target_company)

    # DCN
    model = DCN(features, task='binary')
    run_model(model, 'DCN', train, test, train_model_input, test_model_input, target_candidate)
    model = DCN(features, task='binary')
    run_model(model, 'DCN_', train, test, train_model_input, test_model_input, target_company)
    
    # AFM
    model = AFM(features, task='binary')
    run_model(model, 'AFM', train, test, train_model_input, test_model_input, target_candidate)
    model = AFM(features, task='binary')
    run_model(model, 'AFM_', train, test, train_model_input, test_model_input, target_company)
        
    # NFM
    model = NFM(features, task='binary')
    run_model(model, 'NFM', train, test, train_model_input, test_model_input, target_candidate)
    model = NFM(features, task='binary')
    run_model(model, 'NFM_', train, test, train_model_input, test_model_input, target_company)          
    
    # AutoInt
    model = AutoInt(features, task='binary')
    run_model(model, 'AutoInt', train, test, train_model_input, test_model_input, target_candidate)
    model = AutoInt(features, task='binary')
    run_model(model, 'AutoInt_', train, test, train_model_input, test_model_input, target_company)
    
    # FGCNN - "did not improve"
    model = FGCNN(features, task='binary')
    run_model(model, 'FGCNN', train, test, train_model_input, test_model_input, target_candidate)
    model = FGCNN(features, task='binary')
    run_model(model, 'FGCNN_', train, test, train_model_input, test_model_input, target_company)

    
def run_all_multi_objective_models():
    train, test, train_model_input, test_model_input, features = get_data()
    
    # biPNN
    model = biPNN(features, task='binary')
    run_biobjective_model(model, 'biPNN', train, test, train_model_input, test_model_input)       

    # biDeepFM
    model = biDeepFM(features, task='binary')
    run_biobjective_model(model, 'biDeepFM', train, test, train_model_input, test_model_input)
    
    # biDCN
    model = biDCN(features, task='binary')
    run_biobjective_model(model, 'biDCN', train, test, train_model_input, test_model_input)
    
    # biAFM
    model = biAFM(features, task='binary')
    run_biobjective_model(model, 'biAFM', train, test, train_model_input, test_model_input)
            
    # biNFM
    model = biNFM(features, task='binary')
    run_biobjective_model(model, 'biNFM', train, test, train_model_input, test_model_input)
    
    # biAutoInt
    model = biAutoInt(features, task='binary')
    run_biobjective_model(model, 'biAutoInt', train, test,  train_model_input, test_model_input)
    
    # biFGCNN
    model = biFGCNN(features, task='binary')
    run_biobjective_model(model, 'biFGCNN', train, test, train_model_input, test_model_input)
    

if __name__ == '__main__':
    run()
