import pandas as pd
import os
from datetime import datetime
import requests
import urllib
# import urllib2
from bs4 import BeautifulSoup
from send_email import send_email


def AZ_Path_create(target_path):
    """
    添加新路径
    :param target_path:
    :return:
    """
    if not os.path.exists(target_path):
        os.makedirs(target_path)


def deal_link(title, link_path):
    try:
        wb_info = requests.get(link_path)
        soup = BeautifulSoup(wb_info.text, 'lxml')

        time_source = soup.select('body div["class"="main_left"] div["class"="left-content"] '
                                  'div["class"="Info"] div["class"="time-source"] ')[0]

        news_time = time_source.select('div["class"="time"]')[0].text.replace('年', '-') \
            .replace('月', '-').replace('日', '')
        w_time = datetime.now().strftime('%Y-%m-%d %H:%M')

        # save_day_path = f'{save_path}/{news_time[:10]}'
        save_day_path = f'{save_path}'
        AZ_Path_create(save_day_path)

        data_source = time_source.select('div["class"="source data-source"]')[0]['data-source']
        # print(news_time, data_source)

        save_str_path = f'{save_day_path}/{data_source}'
        # AZ_Path_create(save_str_path)

        main_info = soup.select('body div["class"="main_left"] div["class"="left-content"] div["class"="Body"]')[0]
        main_str = main_info.text.replace('\n', '').replace('\u3000', '').replace('\r', '').replace(' ', '')
        wr_str = '|'.join([title, w_time, news_time, link_path, main_str]) + '\n'
        with open(save_str_path, 'a') as file:
            file.write(wr_str)

    except Exception as error:
        print(error)
        # send_email(text, to, filepath, subject)
        print(title, link_path)
        # send_email(link_path, ['whs@yingpei.com'], [], f'[spider error]{title}')


ind_name = '甲醛'
root_path = f'/home/haishuowang/PycharmProjects/dat_whs/{ind_name}'
save_path = f'/home/haishuowang/PycharmProjects/dat_whs/{ind_name}/temp'

# page_num_list = sorted([int(x.split('.')[0].replace(ind_name, '')) for x in sorted(os.listdir(root_path))])
# # '时间', '标题', '来源', '作者', '网址'
# print(page_num_list)
# for page_num in page_num_list:
#     data = pd.read_excel(f'{root_path}/{ind_name}_财经新闻{page_num}.xls', index_col=0)
#     print(data)
#     data = data[data.index.str.isdigit().fillna(False)]
#     print(data)
#     # for autor, tmp_df in data.groupby(['来源']):
#     #     print(autor, tmp_df)
#     for num, tmp_df in data.iterrows():
#         deal_link(tmp_df['标题'], tmp_df['网址'])

# data = pd.read_excel(f'{root_path}/钢材_财经新闻1.xls', index_col=0)
news_file_list = sorted(os.listdir(root_path))
if 'temp' in news_file_list:
    news_file_list.remove('temp')
for file_name in news_file_list:
    print(file_name)
    data = pd.read_excel(f'{root_path}/{file_name}', index_col=0)
    data = data[data.index.str.isdigit().fillna(False)]
    for num, tmp_df in data.iterrows():
        deal_link(tmp_df['标题'], tmp_df['网址'])
