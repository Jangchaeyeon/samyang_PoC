#####################################################################################
# optimize_pipeline 실행파일
# run_pipeline에 적용되는 전처리 파라미터 & 알고리즘 별 하이퍼파라미터를 최적화하는 실행 코드
# 1. 사이트별 전처리 최적화
# 2. 사이트별 알고리즘 하이퍼파라미터 최적화
# 3. 최적화된 전처리 & 하이퍼파라미터를 이용해 모델 생성
#####################################################################################
# !! 주의 !! py 파일 실행하려면 디렉토리 설정 필요
# configparser  __getitem__ 에러
# configparser(conf_db.ini) 파일을 디렉토리로 설정해줘야 위 에러가 안남
import datetime
import os
import sys
basedir = os.path.dirname(os.path.abspath('C:/sy_packaging/conf_db.ini'))
os.chdir(basedir)
sys.path.append(basedir)

from bayes_opt import BayesianOptimization
import connecting_pipeline_db.optimize_pipeline as optimize_pipeline
import connecting_pipeline_db.run_pipeline as run_pipeline
import pandas as pd
import numpy as np
import configparser
import timeit
import json
import pickle

# configparser로부터 DB 정보 가져오기
conf = configparser.ConfigParser()
conf.read('conf_db.ini', encoding='utf-8')
save_dir = conf['local_dir']['params_dir']

# train 대상 날짜 - 모델 업데이트 날짜로부터 600일 전까지
end_date = datetime.datetime.today() # 파일이 실행되는 날짜,시간
start_date = end_date - datetime.timedelta(days = 600)

end_date = int(str(end_date.date()).replace('-',''))
start_date = int(str(start_date.date()).replace('-',''))

n_iter = 3 # 베이지안 최적화 탐색 iteration 수
# n_iter = 5 기준
# 최적 전처리 탐색 소요시간
#   - 진천 1시간데이터(10분), 15분 데이터(5분) / 대전 1시간 데이터(20분), 15분 데이터(22분) / 대전2 1시간 데이터(9분) / 대전2 15분 데이터(6분)
# 최적 하이퍼파라미터 탐색 소요시간
#   - 진천 1시간데이터(17분), 15분 데이터(12분) / 대전 1시간 데이터(???분), 15분 데이터(분) / 대전2 1시간 데이터(10분) / 대전2 15분 데이터(5분)
first_start = timeit.default_timer()

###################################
# 1. 사이트별 전처리 최적화
###################################

# 사이트별 알고리즘 디폴트 옵션
optimize_list = [########## 진천공장 ############
                {'site_name':'jincheon',
                 'model_info':[{'data_type': 'actual', 'model_name': 'LM'},
                                 {'data_type': 'actual', 'model_name': 'LGBM',
                                  'params': {'max_depth': 20, 'num_leaves': 60, 'max_bin': 30, 'train_iter': 1000}}],
                 'autoregresive_hour':[72]},
                ########## 대전1공장 ###########
                {'site_name':'daejeon1',
                 'model_info':[{'data_type': 'actual', 'model_name': 'XGBOOST',
                                'params':{'n_estimators':250, 'learning_rate' :0.1, 'gamma': 0, 'subsample':0.7, 'colsample_bytree':1, 'max_depth':20}},
                               {'data_type': 'actual', 'model_name': 'LGBM',
                                'params': {'max_depth': 20, 'num_leaves': 60, 'max_bin': 30, 'train_iter': 1000}}],
                 'autoregresive_hour':[72]},
                ########## 대전2공장 ###########
                 {'site_name': 'daejeon2',
                  'model_info': [{'data_type': 'actual', 'model_name': 'LGBM',
                                  'params': {'max_depth': 20, 'num_leaves': 60, 'max_bin': 30, 'train_iter': 1000}}],
                  'autoregresive_hour': [72]}
                 ]

# 위에서 지정한 알고리즘 옵션을 바탕으로 전처리 최적화
# RAM 20G 기준
# 소요시간: 최적화 iter 2기준 -> bo.maximize(n_iter = 2)
#       진천: hour - 514sec / 15min - 282sec
#       대전1: hour = 787sec / 15min - 811sec
#       대전2: hour - 426sec / 15min - 229sec
print('-----------------------------------------------')
print('----------     전처리 옵션 최적화    ------------')
print('-----------------------------------------------')
for opt in optimize_list:
    site_name = opt['site_name']
    model_info = opt['model_info']
    autoregressive_hour = opt['autoregresive_hour']
    print('----------', opt['site_name'], 'start ----------')

    for time_unit_tmp in ['hour','15min']:
        time_unit = time_unit_tmp
        print('----------', time_unit, '----------')
        start = timeit.default_timer()
        # print(model_info)
        # objective_function: 전처리 최적화 대상인 목적함수: validation set의 MAE 반환
        def objective_function(cut_number, B_volume_size, B_weight_size, P_weight_size, model_weight):
            if len(opt['model_info']) > 1: # 알고리즘 두 개 이상인 경우 모델 앙상블 가중치 필요
                model_weight = [model_weight, 1-model_weight]
            else: # 알고리즘 하나인 경우 모델별 앙상블 필요 없음
                model_weight = [1]
            cut_number =  int(cut_number)*25
            B_volume_size = int(B_volume_size)
            B_weight_size = int(B_weight_size)
            P_weight_size = int(P_weight_size)

            loss = optimize_pipeline.get_model_loss(site_name=site_name,  # 'jincheon', 'daejeon2'
                                                     start_date=start_date,  # 훈련데이터 시작일 ex. start_date=20200101
                                                     end_date=end_date,  # 훈련데이터 종료일 ex. end_date=20210227
                                                     model_info=model_info,
                                                     model_weight=model_weight,
                                                     time_unit=time_unit,  # time_unit='15min'
                                                     autoregressive_hour=autoregressive_hour,
                                                     ess_imputation=True, # True로 픽스
                                                     data_filter={'cut_number': cut_number,  # integer ex. 100, 데이터 제거할 숫자, 제거 대상 시간대에 해당하는 행 제거
                                                                  'cut_percentile': None,  # 누적 정규분포 기준 제거 하위,상위 분위수 ex. [0.01, 0.99]
                                                                  'method': None},
                                                     eq_info_include=True,  # True로 픽스, 데이터 병합시 설비정보 포함 여부
                                                     std_mapping_include=True,  # True로 픽스, 데이터 병합시 표준공정조건표 포함 여부
                                                     merge_params={'group_type': 'quantile',
                                                                   'B_volume_size':B_volume_size,
                                                                   'B_weight_size': B_weight_size,
                                                                   'P_weight_size': P_weight_size},
                                                     validation_ratio=0.15,
                                                     validation_seed=2021).calulate_valid_loss()
            return loss # 음수 MAE 반환 <- 파이썬의 베이지안 최적화는 최대화 함수이므로 loss에 음수취해야 loss 최소화됨

        # bo: 베이지안 최적화를 이용해 objective function을 최적화
        #  어떤 전처리 조합에서 validation set의 loss가 최소화되는지 탐색
        if len(opt['model_info'])>1: # 알고리즘 2개 이상인 경우 앙상블, model_weight 탐색 필요
            bo = BayesianOptimization(
                f = objective_function,
                pbounds = {'cut_number':(2,20),
                           'B_volume_size':(3,10),
                           'B_weight_size':(3,10),
                           'P_weight_size':(3,10),
                           'model_weight':(0.2,0.8)},
                random_state = 2021
            )
        else: # 알고리즘 하나인 경우 모델별 앙상블 필요 없음 -> 모델 가중치 [1]
            bo = BayesianOptimization(
                f = objective_function,
                pbounds = {'cut_number':(2,20),
                           'B_volume_size':(3,10),
                           'B_weight_size':(3,10),
                           'P_weight_size':(3,10),
                           'model_weight':(1,1)}, # 모델 가중치 [1]
                random_state = 2021
            )
        bo.maximize(n_iter = n_iter)

        # 최적 전처리 파라미터 탐색결과
        opt_result = pd.DataFrame(bo.res)
        # 최적 결과 추출
        opt_result = opt_result.loc[np.argmax(opt_result['target']), 'params']

        # 최적 결과 저장
        save_file = open(save_dir+'/'+opt['site_name']+'_'+time_unit+'_preprocessed_params.json', "w")
        json.dump(opt_result, save_file)
        save_file.close()

        stop = timeit.default_timer()
        print('Time: ', stop - start)
    print('----------', opt['site_name'], 'complete ----------')

#######################################
# 2. 사이트별 알고리즘 하이퍼파리미터 최적화
#######################################
# 1에서 구한 최적 전처리 파라미터를 고정 후 알고리즘의 하이퍼파라미터 탐색
# RAM 20G 기준
# 소요시간: 최적화 iter 2기준 -> bo.maximize(n_iter = 2)
#       진천: hour - 591sec / 15min - sec
#       대전1: hour - 652sec / 15min - 1104sec
#       대전2: hour - 363sec / 15min - 295sec
print('-----------------------------------------------')
print('---------     하이퍼파라미터 최적화    -----------')
print('-----------------------------------------------')
for opt in optimize_list: # 최적화 대상의 정보가 담긴 리스트 - 공장명, 모델, 자기회귀항
    print('----------', opt['site_name'], 'start ----------')
    for time_unit in ['hour','15min']: # 시간기준 구분
        print('----------', time_unit, '----------')
        start = timeit.default_timer()

        # 전처리 최적 파라미터 가져오기
        with open(save_dir + '/' + opt['site_name'] + '_' + time_unit + '_preprocessed_params.json', "r") as save_file:
            pp_params = json.load(save_file)  # preprocessed parameters
        print(pp_params)
        # pp_params(preprocessed parameters): 최적 'B_volume_size', 'B_weight_size', 'P_weight_size', 'cut_number', 'model_weight'

        # 사이트별로 모델 구성이 달라서 최적화할 하이퍼파라미터가 다름 -> 최적화 목적함수 달라짐
        #------- 진천 목적함수 생성 ------#
        # 적용 알고리즘: LM & LightGBM
        # LightGBM의 하이퍼파라미터 탐색
        if opt['site_name'] == 'jincheon':
            site_name = opt['site_name']
            model_info = opt['model_info']
            autoregressive_hour = opt['autoregresive_hour']

            # 진천: LM+LGBM 조합 -> LGBM 파라미터만 조정하면 됨
            def objective_function(lgbm_max_depth, lgbm_num_leaves, lgbm_max_bin):
                opt['model_info'][1]['params'] =  {'max_depth': int(lgbm_max_depth),
                                                   'num_leaves': int(lgbm_num_leaves),
                                                   'max_bin': int(lgbm_max_bin), 'train_iter': 1500}
                # 전처리 최적 파라미터 고정
                if len(opt['model_info']) > 1: # 알고리즘 두 개 이상인 경우 모델 앙상블 가중치 필요
                    model_weight = [pp_params['model_weight'], 1-pp_params['model_weight']]
                else: # 알고리즘 하나인 경우 모델별 앙상블 필요 없음
                    model_weight = [1]
                cut_number =  int(pp_params['cut_number'])*25
                B_volume_size = int(pp_params['B_volume_size'])
                B_weight_size = int(pp_params['B_weight_size'])
                P_weight_size = int(pp_params['P_weight_size'])

                # 알고리즘의 하이퍼파라미터에 따른 validation loss 구하기
                loss = optimize_pipeline.get_model_loss(site_name=site_name,  # 'jincheon', 'daejeon2'
                                                         start_date=start_date,  # 훈련데이터 시작일 ex. start_date=20200101
                                                         end_date=end_date,  # 훈련데이터 종료일 ex. end_date=20210227
                                                         model_info=model_info,
                                                         model_weight=model_weight,
                                                         time_unit=time_unit,  # time_unit='15min'
                                                         autoregressive_hour=autoregressive_hour,
                                                         ess_imputation=True, # True로 픽스
                                                         data_filter={'cut_number': cut_number,  # integer ex. 100, 데이터 제거할 숫자, 제거 대상 시간대에 해당하는 행 제거
                                                                      'cut_percentile': None,  # 누적 정규분포 기준 제거 하위,상위 분위수 ex. [0.01, 0.99]
                                                                      'method': None},
                                                         eq_info_include=True,  # True로 픽스, 데이터 병합시 설비정보 포함 여부
                                                         std_mapping_include=True,  # True로 픽스, 데이터 병합시 표준공정조건표 포함 여부
                                                         merge_params={'group_type': 'quantile',
                                                                       'B_volume_size':B_volume_size,
                                                                       'B_weight_size': B_weight_size,
                                                                       'P_weight_size': P_weight_size},
                                                         validation_ratio=0.15,
                                                         validation_seed=2021).calulate_valid_loss()
                return loss # 음수 MAE 반환 <- 파이썬의 베이지안 최적화는 최대화 함수이므로 loss에 음수취해야 loss 최소화됨

            bo = BayesianOptimization(
                f=objective_function,
                pbounds={'lgbm_max_depth':(10,30),
                         'lgbm_num_leaves':(60,150),
                         'lgbm_max_bin':(20,50)},
                random_state=2021
            )
            bo.maximize(n_iter=n_iter)

        #------- 대전1 목적함수 생성 ------#
        # 적용 알고리즘: XGBOOST & LightGBM
        # XGBOOST, LightGBM의 하이퍼파라미터 탐색
        elif opt['site_name'] == 'daejeon1':
            site_name = opt['site_name']
            model_info = opt['model_info']
            autoregressive_hour = opt['autoregresive_hour']

            # 대전1: XGBOOST+LGBM 조합 -> 둘 다 조절해야 함
            def objective_function(xgboost_n_estimators, xgboost_max_depth, lgbm_max_depth, lgbm_num_leaves, lgbm_max_bin):
                opt['model_info'][0]['params'] = {'n_estimators': int(xgboost_n_estimators),
                                                  'learning_rate' :0.1,
                                                  'gamma': 0,
                                                  'subsample':0.7,
                                                  'colsample_bytree':1,
                                                  'max_depth':int(xgboost_max_depth)}
                opt['model_info'][1]['params'] = {'max_depth': int(lgbm_max_depth),
                                                  'num_leaves': int(lgbm_num_leaves),
                                                  'max_bin': int(lgbm_max_bin), 'train_iter': 1500}
                if len(opt['model_info']) > 1:  # 알고리즘 두 개 이상인 경우 모델 앙상블 가중치 필요
                    model_weight = [pp_params['model_weight'], 1 - pp_params['model_weight']]
                else:  # 알고리즘 하나인 경우 모델별 앙상블 필요 없음
                    model_weight = [1]
                cut_number = int(pp_params['cut_number']) * 25
                B_volume_size = int(pp_params['B_volume_size'])
                B_weight_size = int(pp_params['B_weight_size'])
                P_weight_size = int(pp_params['P_weight_size'])

                loss = optimize_pipeline.get_model_loss(site_name=site_name,  # 'jincheon', 'daejeon1','daejeon2'
                                                        start_date=start_date,  # 훈련데이터 시작일 ex. start_date=20200101
                                                        end_date=end_date,  # 훈련데이터 종료일 ex. end_date=20210227
                                                        model_info=model_info,
                                                        model_weight=model_weight,
                                                        time_unit=time_unit,  # time_unit='15min'
                                                        autoregressive_hour=autoregressive_hour,
                                                        ess_imputation=True,  # True로 픽스
                                                        data_filter={'cut_number': cut_number,
                                                                     # integer ex. 100, 데이터 제거할 숫자, 제거 대상 시간대에 해당하는 행 제거
                                                                     'cut_percentile': None,
                                                                     # 누적 정규분포 기준 제거 하위,상위 분위수 ex. [0.01, 0.99]
                                                                     'method': None},
                                                        eq_info_include=True,  # True로 픽스, 데이터 병합시 설비정보 포함 여부
                                                        std_mapping_include=True,  # True로 픽스, 데이터 병합시 표준공정조건표 포함 여부
                                                        merge_params={'group_type': 'quantile',
                                                                      'B_volume_size': B_volume_size,
                                                                      'B_weight_size': B_weight_size,
                                                                      'P_weight_size': P_weight_size},
                                                        validation_ratio=0.15,
                                                        validation_seed=2021).calulate_valid_loss()
                return loss  # 음수 MAE 반환 <- 파이썬의 베이지안 최적화는 최대화 함수이므로 loss에 음수취해야 loss 최소화됨

            bo = BayesianOptimization(
                f=objective_function,
                pbounds={'xgboost_n_estimators':(200,400),
                         'xgboost_max_depth':(10,40),
                         'lgbm_max_depth': (10, 30),
                         'lgbm_num_leaves': (60,150),
                         'lgbm_max_bin': (20, 50)},
                random_state=2021
            )
            bo.maximize(n_iter=n_iter)

        # ------- 대전2 목적함수 생성 ------#
        # 적용 알고리즘: LightGBM
        # LightGBM의 하이퍼파라미터 탐색
        else:
            site_name = opt['site_name']
            model_info = opt['model_info']
            autoregressive_hour = opt['autoregresive_hour']

            def objective_function(lgbm_max_depth, lgbm_num_leaves, lgbm_max_bin):
                opt['model_info'][0]['params'] = {'max_depth': int(lgbm_max_depth),
                                                  'num_leaves': int(lgbm_num_leaves),
                                                  'max_bin': int(lgbm_max_bin), 'train_iter': 1500}
                if len(opt['model_info']) > 1:  # 알고리즘 두 개 이상인 경우 모델 앙상블 가중치 필요
                    model_weight = [pp_params['model_weight'], 1 - pp_params['model_weight']]
                else:  # 알고리즘 하나인 경우 모델별 앙상블 필요 없음
                    model_weight = [1]
                cut_number = int(pp_params['cut_number']) * 25
                B_volume_size = int(pp_params['B_volume_size'])
                B_weight_size = int(pp_params['B_weight_size'])
                P_weight_size = int(pp_params['P_weight_size'])

                loss = optimize_pipeline.get_model_loss(site_name=site_name,
                                                        start_date=start_date,
                                                        end_date=end_date,
                                                        model_info=model_info,
                                                        model_weight=model_weight,
                                                        time_unit=time_unit,
                                                        autoregressive_hour=autoregressive_hour,
                                                        ess_imputation=True,
                                                        data_filter={'cut_number': cut_number,
                                                                     'cut_percentile': None,
                                                                     'method': None},
                                                        eq_info_include=True,
                                                        std_mapping_include=True,
                                                        merge_params={'group_type': 'quantile',
                                                                      'B_volume_size': B_volume_size,
                                                                      'B_weight_size': B_weight_size,
                                                                      'P_weight_size': P_weight_size},
                                                        validation_ratio=0.15,
                                                        validation_seed=2021).calulate_valid_loss()
                return loss

            bo = BayesianOptimization(
                f=objective_function,
                pbounds={'lgbm_max_depth':(10,30),
                         'lgbm_num_leaves':(60,150),
                         'lgbm_max_bin':(20,50)},
                random_state=2021
            )
            bo.maximize(n_iter=n_iter)

        # 사이트/시간단위 모델 별 최적 전처리 파라미터 탐색결과
        opt_result = pd.DataFrame(bo.res)
        # 최적 결과 추출
        opt_result = opt_result.loc[np.argmax(opt_result['target']), 'params']

        # 최적 결과 저장
        save_file = open(save_dir + '/' + opt['site_name'] + '_' + time_unit + '_hyperparams.json', "w")
        json.dump(opt_result, save_file)
        save_file.close()

        stop = timeit.default_timer()
        print('Time: ', stop - start)
    print('----------', opt['site_name'], 'complete ----------')


##################################################
# 3. 최적화된 전처리 & 하이퍼파라미터를 이용해 모델 생성
##################################################
print('-----------------------------------------------')
print('--------------     모델 생성    ----------------')
print('-----------------------------------------------')
for site_name in ['jincheon','daejeon1', 'daejeon2']:
    print('----------', site_name, 'start ----------')
    for time_unit in ['hour','15min']: # 사이트/시간데이터 별 구분
        print('----------', time_unit, '----------')
        # 전처리 파라미터 가져오기
        with open(save_dir + '/' + site_name + '_' + time_unit + '_preprocessed_params.json', "r") as save_file:
            pp_params = json.load(save_file)  # preprocessed parameters
        # print(pp_params)
        # 모델 별 하이퍼파라미터 가져오기
        with open(save_dir + '/' + site_name + '_' + time_unit + '_hyperparams.json', "r") as save_file:
            h_params = json.load(save_file)  # preprocessed parameters
        # print(h_params)

        # 사이트별 모델 정보 가져오기
        if site_name == 'jincheon':
            model_info = [{'data_type': 'actual', 'model_name': 'LM'},
                          {'data_type': 'actual', 'model_name': 'LGBM',
                           'params': {'max_depth': int(h_params['lgbm_max_depth']),
                                      'num_leaves': int(h_params['lgbm_num_leaves']),
                                      'max_bin': int(h_params['lgbm_max_bin']), 'train_iter': 1500}}]
        elif site_name == 'daejeon1':
            model_info = [{'data_type': 'actual', 'model_name': 'XGBOOST',
                             'params': {'n_estimators': int(h_params['xgboost_n_estimators']), 'learning_rate': 0.1, 'gamma': 0, 'subsample': 0.7,
                                        'colsample_bytree': 1, 'max_depth': int(h_params['xgboost_max_depth'])}},
                            {'data_type': 'actual', 'model_name': 'LGBM',
                             'params': {'max_depth': int(h_params['lgbm_max_depth']),
                                      'num_leaves': int(h_params['lgbm_num_leaves']),
                                      'max_bin': int(h_params['lgbm_max_bin']), 'train_iter': 1500}}]
        else:
            model_info =[{'data_type': 'actual', 'model_name': 'LGBM',
                           'params': {'max_depth': int(h_params['lgbm_max_depth']),
                                      'num_leaves': int(h_params['lgbm_num_leaves']),
                                      'max_bin': int(h_params['lgbm_max_bin']), 'train_iter': 1500}}]

        if len(model_info) > 1:  # 알고리즘 두 개 이상인 경우 모델 앙상블 가중치 필요
            model_weight = [pp_params['model_weight'], 1 - pp_params['model_weight']]
        else:  # 알고리즘 하나인 경우 모델별 앙상블 필요 없음
            model_weight = [1]

        obj = run_pipeline.train_model( site_name = site_name,
                                        start_date = start_date,
                                        end_date = end_date,
                                        model_info = model_info,
                                        model_weight = model_weight,
                                        time_unit = time_unit,
                                        autoregressive_hour = [72], # 자기회귀항 시간
                                        ess_imputation=True, # ESS 결측 채울지 여부
                                        # capa 기준 데이터 필터링
                                        data_filter={'cut_number': int(pp_params['cut_number']) * 25,
                                                     'cut_percentile': None,
                                                     'method': None},
                                        eq_info_include=True,  # 데이터 병합시 설비정보 포함 여부
                                        std_mapping_include=True,  # 데이터 병합시 표준공정조건표 포함 여부
                                        merge_params={'group_type': 'quantile',
                                                      'B_volume_size': int(pp_params['B_volume_size']),
                                                      'B_weight_size': int(pp_params['B_weight_size']),
                                                      'P_weight_size': int(pp_params['P_weight_size'])},
                                        validation_ratio = 0.15,
                                        validation_seed = 2021)
        with open(save_dir+'/' + site_name + '_' + time_unit + '_' + 'model_object.pkl', 'wb') as outp:
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    print('----------', site_name, 'complete ----------')

last_stop = timeit.default_timer()
print('Time: ', last_stop - first_start)

#####################################################################################
# 모델링 제품 그룹 경계값 추출
#####################################################################################
params_dat = pd.DataFrame()
for site_name  in ['jincheon','daejeon1', 'daejeon2']:
    for time_unit in ['hour','15min'] :
        print('--------------',site_name,'&',time_unit, '--------------')
        with open(save_dir+'/' + site_name + '_' + time_unit + '_' + 'model_object.pkl', 'rb') as inp:
            obj = pickle.load(inp)
        temp_obj = obj.get_train_data()
        temp_obj_bins = temp_obj[0]['train_set_bins']
        with open(save_dir + '/' + site_name + '_' + time_unit + '_preprocessed_params.json', "r") as save_file:
            pp_params = json.load(save_file)
        # dataframe 만들기
        temp_df = pd.DataFrame(columns=['사이트', '시간단위'])
        temp_df['사이트'] = [site_name, site_name, site_name]
        temp_df['시간단위'] = [time_unit, time_unit, time_unit]
        temp_df['구분'] = ["Bottle_volume", "Bottle_weight", 'Preform_weight']
        temp_df['그룹개수'] = [int(pp_params['B_volume_size']), int(pp_params['B_weight_size']),
                           int(pp_params['P_weight_size'])]  # , "Bottle_weight", 'Preform_weight'
        temp_df['경계값'] = [str(list(temp_obj_bins['b_v_boundary'])), str(list(temp_obj_bins['b_w_boundary'])),
                          str(list(temp_obj_bins['p_w_boundary']))]
        params_dat =  pd.concat([params_dat, temp_df]).reset_index(drop=True)

        params_dat.to_excel(save_dir + "/사이트_모델별_제품그룹경계값.xlsx")
