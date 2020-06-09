import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import re

date_now = datetime.now()
#
# data = pd.read_csv('/home/haishuowang/spider_data/2019-07-17/兰格钢铁网', sep='|', header=None)
# data.columns = ['Title', 'w_time', 'n_time', 'Link', 'Info']
# data = data[~data['Title'].duplicated(keep='first')]
#
# # x =
# data['deal_title'] = data.apply(lambda x: x['Title'].replace(f'{int(date_now.month)}月', '')
#                                 .replace(f'{int(date_now.day)}日', '')
#                                 , axis=1)
#
mid_word = ['稳', '→', '震荡', '平', ]
buy_word = ['涨', '上调', '↑', '上行', '强势', '走高']
sell_word = ['跌', '降', '下调', '探低', '↓', '下行', '弱势', '走低']
# # 方大特钢
#
#


file_name_list = ['全球金属网', '兰格钢铁网', '大宗内参', '海鑫钢网', '瑞达期货', '生意社', '西本新干线']


def contain(x, key_word, label=1):
    for key in key_word:
        if key in x:
            return label
    else:
        return np.nan


def load_spot_data(read_path):
    file_data = pd.read_csv(read_path, sep='|', header=None)
    file_data.columns = ['Title', 'w_time', 'n_time', 'Link', 'Info']
    file_data.index = pd.to_datetime(file_data['n_time'])
    return file_data


def filer_target_word(raw_df):
    target_df = raw_df[raw_df['Title'].str.contains('钢')]
    return target_df


def get_file_pos(file_name):
    root_path = '/home/haishuowang/temp'
    date_list = sorted(os.listdir(root_path))
    # file_name = '兰格钢铁网'

    data_list = []
    for target_date in date_list:
        read_path = f'{root_path}/{target_date}/{file_name}'
        if os.path.exists(f'{root_path}/{target_date}/{file_name}'):
            file_data = pd.read_csv(read_path, sep='|', header=None)
            file_data.columns = ['Title', 'w_time', 'n_time', 'Link', 'Info']
            file_data.index = pd.to_datetime(file_data['n_time']) + timedelta(minutes=10)
            file_data = file_data.sort_index()

            mid = file_data['Title'].apply(lambda x: contain(x, mid_word, label=0))
            mid.name = 'mid'
            buy = file_data['Title'].apply(lambda x: contain(x, buy_word, label=1))
            buy.name = 'buy'
            sell = file_data['Title'].apply(lambda x: contain(x, sell_word, label=-1))
            sell.name = 'sell'

            mid_info = file_data['Info'].apply(lambda x: contain(x, mid_word, label=0))
            mid_info.name = 'mid_info'
            buy_info = file_data['Info'].apply(lambda x: contain(x, buy_word, label=1))
            buy_info.name = 'buy_info'
            sell_info = file_data['Info'].apply(lambda x: contain(x, sell_word, label=-1))
            sell_info.name = 'sell_info'

            # no_info = mid_info.isna() & buy_info.isna() & sell_info.isna()
            part_info = pd.concat([file_data['Title'], mid, buy, sell, mid_info, buy_info, sell_info], axis=1)
            data_list.append(part_info)
        else:
            print(target_date)
            pass

    all_info = pd.concat(data_list, axis=0)
    all_info.to_csv(f'/home/haishuowang/PycharmProjects/{file_name}.csv')
    return all_info


def get_spider_file_pos(file_name='生意社'):
    root_path = '/home/haishuowang/spider_data'
    date_list = sorted([x for x in os.listdir(root_path) if len(x) == 10 and '-' in x and x > '2019-07-18'])
    data_list = []
    for target_date in date_list:
        read_path = f'/home/haishuowang/spider_data/{target_date}/{file_name}'
        if os.path.exists(f'{root_path}/{target_date}/{file_name}'):
            file_data = load_spot_data(read_path)
            file_data = filer_target_word(file_data)

            file_data.index = pd.to_datetime(file_data['n_time']) + timedelta(minutes=10)
            file_data = file_data.sort_index()

            mid = file_data['Title'].apply(lambda x: contain(x, mid_word, label=0))
            mid.name = 'mid'
            buy = file_data['Title'].apply(lambda x: contain(x, buy_word, label=1))
            buy.name = 'buy'
            sell = file_data['Title'].apply(lambda x: contain(x, sell_word, label=-1))
            sell.name = 'sell'

            mid_info = file_data['Info'].apply(lambda x: contain(x, mid_word, label=0))
            mid_info.name = 'mid_info'
            buy_info = file_data['Info'].apply(lambda x: contain(x, buy_word, label=1))
            buy_info.name = 'buy_info'
            sell_info = file_data['Info'].apply(lambda x: contain(x, sell_word, label=-1))
            sell_info.name = 'sell_info'

            part_info = pd.concat([file_data['Title'], mid, buy, sell, mid_info, buy_info, sell_info], axis=1)
            data_list.append(part_info)
        else:
            print(target_date)
            pass

    all_info = pd.concat(data_list, axis=0)
    all_info.to_csv(f'/home/haishuowang/PycharmProjects/{file_name}_spider.csv')
    return all_info


# data_0717 = load_spot_data('/home/haishuowang/spider_data/2019-07-17/生意社')
# data_0719 = load_spot_data('/home/haishuowang/spider_data/2019-07-19/生意社')
# # data_0720 = load_spot_data('/home/haishuowang/spider_data/2019-07-20/生意社')
# data_0722 = load_spot_data('/home/haishuowang/spider_data/2019-07-22/生意社')
# data_0723 = load_spot_data('/home/haishuowang/spider_data/2019-07-23/生意社')
#
# data_0717 = filer_target_word(data_0717)
# data_0719 = filer_target_word(data_0719)
# data_0722 = filer_target_word(data_0722)
# data_0723 = filer_target_word(data_0723)

# all_info = get_spider_file_pos(file_name='生意社')

# for file_name in file_name_list:
#     all_info = get_file_pos(file_name)


# if __name__ == '__main__':


def deal_jd_data(fut_name='鸡蛋', file_name='金谷高科'):
    root_path = f'/home/haishuowang/PycharmProjects/dat_whs/{fut_name}/temp'
    target_date_list = sorted([x for x in os.listdir(root_path) if x >= '2019-06-25'])
    print(target_date_list)
    result_list = []
    for target_date in target_date_list:
        if os.path.exists(f'{root_path}/{target_date}/{file_name}'):
            print('_______________')
            print(target_date)
            info_data = load_spot_data(f'{root_path}/{target_date}/{file_name}')
            print(info_data)
            # print(info_data[info_data['Title'].str.contains('辽宁')])
            # result_list.append(info_data[info_data['Title'].str.contains('辽宁')])
        else:
            pass
    return pd.concat(result_list, axis=0)


def deal_cf_data(fut_name='棉花', file_name='金谷高科'):
    root_path = f'/home/haishuowang/PycharmProjects/dat_whs/{fut_name}/temp'
    info_data = load_spot_data(f'{root_path}/{file_name}')
    return info_data.sort_index().drop_duplicates()


# '/home/haishuowang/PycharmProjects/dat_whs/甲醛/temp/生意社'

# info_data = deal_cf_data(fut_name='甲醛', file_name='生意社')

fut_name = '甲醛'
file_name = '生意社'
info_data = deal_cf_data(fut_name, file_name)


def title_filter(info_sr):
    print(info_sr)
    title = info_sr.iloc[0]
    print(info_sr.name)
    month_t, day_t = pd.to_datetime(info_sr.name).strftime('%m/%d').split('/')
    print(title, month_t, day_t)
    date_str = f'{str(int(month_t))}月{str(int(day_t))}日'
    if date_str in title:
        return True
    else:
        return False


info_data = info_data[info_data[['Title']].apply(title_filter, axis=1)]

mid = info_data['Title'].apply(lambda x: contain(x, mid_word, label=0))
mid.name = 'mid'
buy = info_data['Title'].apply(lambda x: contain(x, buy_word, label=1))
buy.name = 'buy'
sell = info_data['Title'].apply(lambda x: contain(x, sell_word, label=-1))
sell.name = 'sell'

mid_info = info_data['Info'].apply(lambda x: contain(x, mid_word, label=0))
mid_info.name = 'mid_info'
buy_info = info_data['Info'].apply(lambda x: contain(x, buy_word, label=1))
buy_info.name = 'buy_info'
sell_info = info_data['Info'].apply(lambda x: contain(x, sell_word, label=-1))
sell_info.name = 'sell_info'

part_info = pd.concat([info_data['Title'], mid, buy, sell, mid_info, buy_info, sell_info], axis=1)
part_info['pos_1'] = part_info[['mid', 'buy', 'sell']].sum(1)
part_info['pos_2'] = part_info[['mid', 'buy', 'sell', 'mid_info', 'buy_info', 'sell_info']].sum(1)
part_info.to_csv(f'~/PycharmProjects/dat_whs/{fut_name}_{file_name}.csv', sep='|')
# result_list = []
# for n_time, part_a in a.iterrows():
#     city_info_list = part_a['Info'].split('：')[1].split('。')[:-1]
#     for city_info in city_info_list:
#         # print(city_info)
#         city_name = city_info.split('市')[0]
#         price_info = city_info.split('市')[1].split('，')[0]
#         price_num = float(re.findall('(?<=为).*?(?=元)', price_info)[0])
#         print(n_time, city_name, price_num)
#         result_list.append([n_time, city_name, price_num])
#
# result_info = pd.DataFrame(result_list, columns=['n_time', 'city_name', 'price_num']).set_index(['n_time', 'city_name'])
# print(result_info.xs('大连', level=1))
