#####################################################################################
# run_pipeline 실행파일
# 1. 프로시저실행 - PRODUCT(1 HOUR) ESS(1 DAY)

# 2. 날씨 - 1DAY
# 2. 순생산량 계산 및 DB 적재
# 3. 사이트별 파이프라인 실행(전처리->병합->모델링)
#####################################################################################
import pandas as pd
import os
import logging
import configparser
import pickle
import datetime
import warnings
import sys
import pyodbc

basedir = os.path.dirname(os.path.abspath('C:/sy_packaging/conf_db.ini'))
os.chdir(basedir)

sys.path.append(basedir)

import preprocessing_db.clean_weather as clean_weather
import preprocessing_db.update_preprocessed_product as update_preprocessed_product
import preprocessing_db.collecting_data as collecting_data
import calculating_target.output_table as output_table
import calculating_target.send_mail as send_mail
import running_db.db_dml as db_dml

# 옵션 : warning 무시
warnings.simplefilter(action='ignore', category=Warning)
# 에러 로그기록
logging.basicConfig(filename= basedir + '/exe_run_pipeline.log',
                    level=logging.ERROR)

# !! 주의 !! py 파일 실행하려면 디렉토리 설정 필요
# configparser  __getitem__ 에러
# configparser(conf_db.ini) 파일을 디렉토리로 설정해줘야 위 에러가 안남
# basedir = os.path.dirname(os.path.abspath('C:/Users/syc720202/PycharmProjects/sy_packaging_cowork/sy_packaging/conf_db.ini'))

#######################################
# 파라미터 설정
#######################################
test_datetime = datetime.datetime.today() # 파일이 실행되는 날짜,시간

test_date = int(test_datetime.strftime("%Y%m%d"))             # 날짜
test_hour = str(test_datetime.hour)                           # 시간
test_datetime = test_datetime.strftime("%Y-%m-%d %H:%M:%S")   # 날짜 시간 형변환

test_days = 5
max_SOC = 0.9

# start_date = 20200115
# end_date = 20210227
# test_date = 20210306
# site_name = "jincheon"
# time_unit = "hour"

# configparser로부터 DB 정보 가져오기
conf = configparser.ConfigParser()
conf.read('conf_db.ini', encoding='utf-8')

def is_work_day():  # workday 면 1이상을 반환(통상4)
    workday = False

    # DB 정보
    db_ver = str(1)
    server = conf['db_info' + db_ver]['server']
    database = conf['db_info' + db_ver]['database']
    username = conf['db_info' + db_ver]['username']
    password = conf['db_info' + db_ver]['password']

    conn = pyodbc.connect("DRIVER={SQL Server};SERVER=" + server + ";uid=" + username + ";pwd=" + password + ";DATABASE=" + database)

    workday_check_query = f""" SELECT COUNT(*) AS 'COUNT' FROM TB_BDC_WORKDAY WHERE DT ='{datetime.datetime.today().strftime('%Y-%m-%d')}'"""
    results = pd.read_sql(workday_check_query, conn)
    workday_count = results.iloc[0]['COUNT']
    if workday_count > 0:
        workday = True

    conn.close()

    return workday

#######################################
# 1. 데이터 적재
#######################################
print('-----------------------------------------------')
print('-------------(외부) 데이터 적재    ---------------')
print('-----------------------------------------------')
# hourly 실행 process - Raw product 프로시저 실행
collecting_data.product()
# daily 실행 process
if test_hour == "1": # 매일 오전 1시 실행
    # ESS collect 실행
    collecting_data.ess()
# if str(test_date)[4:6] in ['07','08','09','12',"01","02"] : #peak 관리 月
if test_hour == "2": # 매일 오전 2시 실행
    # 한전 peak of year
    file_path = conf['local_dir']['elec_peak_dir']
    collecting_data.hj_peak(save_path = file_path)
if test_hour == "3": # 매일 오전 3시 실행
    for site_name in ['jincheon', 'daejeon']:
        clean_weather.crawl_forecast(site_name, save=True)

print('-----------------------------------------------')
print('-----------     테스트 데이터 적합    ------------')
print('-----------------------------------------------')
# conf = configparser.ConfigParser()
# conf.read('conf_db.ini', encoding='utf-8')
save_dir = conf['local_dir']['params_dir'] # 모델 최적화 결과 저장 경로

#for site_name in ['daejeon1']:
for site_name in ['jincheon','daejeon1', 'daejeon2']:
    print(" ")
    print('----------Temp site : ', site_name, '----------')
    ###################################
    # 2. 순생산량 계산 및 DB 적재
    ###################################
    # - actual # 진천 17sec
    # start = timeit.default_timer()
    update_preprocessed_product.update(site_name=site_name,  # 'jincheon', 'daejeon1', 'daejoen2'
                                       today_date=test_date,    # 8자리 int 날짜. ex) 20211212
                                       tb_type="actual")   # 'actual'(생산실적), 'plan'(생산계획)
    # stop = timeit.default_timer()
    # # print('Actual Product Update Time: ', stop - start)
    #
    # # - plan # 진천 5sec
    # start = timeit.default_timer()
    update_preprocessed_product.update(site_name=site_name,  # 'jincheon', 'daejeon1', 'daejoen2'
                                       today_date=test_date,    # 8자리 int 날짜. ex) 20211212
                                       tb_type="plan")     # 'actual'(생산실적), 'plan'(생산계획)
    # stop = timeit.default_timer()
    # print('Plan Product Update Time: ', stop - start)

    ###################################
    # 3. 사이트별 파이프라인 실행(전처리->병합->모델링)
    ###################################
    # 오브젝트 생성
    # 오브젝트 안에 저장되는 것:
    #   - 훈련 데이터
    #       - 'data_type': 훈련데이터 종류-plan or actual
    #       - 'train'/'valid' <-훈련데이터를 랜덤샘플링
    #       - 'train_set_bins' <-quantile 방식으로 전처리했을 시 저장
    #   - 훈련된 모델 리스트

    # for time_unit in ['hour','15min']: # 사이트/시간데이터 별 구분
    for time_unit in ['hour']:  # 사이트/시간데이터 별 구분, 15min 일단 제외 (221025)
        print('---', time_unit)
        with open(save_dir+'/' + site_name + '_' + time_unit + '_' + 'model_object.pkl', 'rb') as inp:
            obj = pickle.load(inp)

        # # test 데이터 예측
        result = obj.predict_test(test_date=test_date, test_days=test_days, test_hour=test_hour, max_SOC=max_SOC)
        # print(result)

        ## 결과 저장
        if time_unit == "hour":
            exec("result_" + site_name + "= result")
        # - save : TB_BDA_PRED_HOURLY, TB_BDA_PRED_15MIN, TB_BDA_MODEL_EVAL
        output_table.output_table(batch_datetime=test_datetime,
                                  batch_date=test_date,
                                  batch_hour=test_hour,
                                  site_name=site_name,
                                  time_unit=time_unit,
                                  result=result)

print("-- total dr")
# - save : TB_BDA_PRED_TOTAL_DR
output_table.output_table_total(dj1_result=result_daejeon1,
                                dj2_result=result_daejeon2,
                                jc_rescult=result_jincheon,
                                batch_datetime=test_datetime,
                                batch_date=test_date)

# - save : TB_BDA_PRED_HOURLY, TB_BDA_MODEL_EVAL 에 통합DR 업데이트
db_dml.tb_bda_model_eval_total_update(test_datetime)

# # 매일 13시마다 메일 발송
# if test_hour == "13":
#     if len(result['result_plan_performance']) > 0 and len(result['result_actual_performance']) > 0:
#         send_mail.send_mail()
#     # if datetime.datetime.today().weekday() != 5 and datetime.datetime.today().weekday() != 6:  # dr참여 결과 전송은 주말 제외
#     if is_work_day():  # workday 인 경우에 만 메일 발송
#         send_mail.send_mail_dr('1300')  # 전체
#         #send_mail.send_mail_dr('1302')  # 진천
#         #send_mail.send_mail_dr('1303')  # 대전1
#         #send_mail.send_mail_dr('1304')  # 대전2