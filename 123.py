#!coding=utf-8
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
# chrome_options.add_argument('--headless')

import pickle
from multiprocessing.dummy import Pool as ThreadPool
import requests
import urllib
# import urllib2
from bs4 import BeautifulSoup
import pandas as pd


class BuffSpider(object):
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.save_path = '/home/haishuowang/spider_data/buff_spider'
        self.driver = webdriver.Chrome('/home/haishuowang/PycharmProjects/nl2sql_baseline-python3/data/chromedriver',
                                       options=chrome_options)
        self.driver.get(url='https://buff.163.com/')
        self.cookies = pickle.load(open("cookies.pkl", "rb"))
        self.set_cookie()
        self.is_login()

    def is_login(self):
        """判断当前是否登陆"""
        self.driver.refresh()
        html = self.driver.page_source
        if html.find(self.username) == -1:  # 利用用户名判断是否登陆
            # 没登录 ,则手动登录
            self.login()
        else:
            # 已经登录  尝试访问搜索记录，可以正常访问
            # self.driver.get(url='http://i.baidu.com/my/history')
            self.driver.get(url='https://buff.163.com/market/steam_inventory?game=dota2#page_num=1&page_'
                                'size=50&search=&state=tradable')
            time.sleep(10)  # 延时看效果

    def login(self):
        """登陆"""
        time.sleep(60)  # 等待手动登录
        self.driver.refresh()
        self.save_cookie()

    def save_cookie(self):
        """保存cookie"""
        # 将cookie序列化保存下来
        pickle.dump(self.driver.get_cookies(), open("cookies.pkl", "wb"))

    def set_cookie_driver(self, driver):
        try:
            for cookie in self.cookies:
                cookie_dict = {
                    "domain": "buff.163.com",  # 火狐浏览器不用填写，谷歌要需要
                    'name': cookie.get('name'),
                    'value': cookie.get('value'),
                    "expires": "",
                    'path': '/',
                    'httpOnly': False,
                    'HostOnly': False,
                    'Secure': False}
                driver.add_cookie(cookie_dict)
        except Exception as e:
            print(e)

    def set_cookie(self):
        """往浏览器添加cookie"""
        """利用pickle序列化后的cookie"""
        try:
            # cookies = pickle.load(open("cookies.pkl", "rb"))
            for cookie in self.cookies:
                cookie_dict = {
                    "domain": "buff.163.com",  # 火狐浏览器不用填写，谷歌要需要
                    'name': cookie.get('name'),
                    'value': cookie.get('value'),
                    "expires": "",
                    'path': '/',
                    'httpOnly': False,
                    'HostOnly': False,
                    'Secure': False}
                self.driver.add_cookie(cookie_dict)
        except Exception as e:
            print(e)

    def get_soup(self, url):
        driver = webdriver.Chrome('/home/haishuowang/PycharmProjects/nl2sql_baseline-python3/data/chromedriver',
                                  options=chrome_options)
        driver.get(url)
        self.set_cookie_driver(driver)
        driver.refresh()
        time.sleep(3)
        # driver.implicitly_wait(10)
        data = driver.page_source
        driver.close()
        return BeautifulSoup(data, 'lxml')

    def get_goods_info(self, part_url):
        root_path = 'https://buff.163.com'
        # part_url = '/market/goods?goods_id=772678'

        sell_str = '#tab=selling'
        buy_str = '#tab=buying'
        history_str = '#tab=history'
        chart_str = '#tab=price-chart'

        for target_str in [sell_str, buy_str]:
            print('*' * 20)
            print(target_str)

            for i in range(1, 11):

                if i == 1:
                    target_url = f'{root_path}/{part_url}{target_str}'
                else:
                    target_url = f'{root_path}/{part_url}{target_str}&page_num={i}'

                print(target_url)
                # self.driver.get(target_url)
                #
                # time.sleep(5)
                # data = self.driver.page_source
                # goods_soup = BeautifulSoup(data, 'lxml')
                goods_soup = self.get_soup(target_url)
                goods_table = goods_soup.select('body div[class="market-list"] div[class="l_Layout"] '
                                                'div[class="detail-tab-cont"] table[class="list_tb"] '
                                                'tbody[class="list_tb_dota2"]')[0]

                for j in range(1, len(goods_table.find_all('tr'))):
                    tr = goods_table.find_all('tr')[j]
                    goods_name = tr.find_all('td')[2].text.replace('\n', '').replace(' ', '')
                    user_name = tr.find_all('td')[3].text.replace('\n', '').replace(' ', '')
                    send_method = tr.find_all('td')[4].text.replace('\n', '').replace(' ', '')
                    price_str = tr.find_all('td')[5].text.replace('\n', '').replace(' ', '').replace('￥', '')
                    price_int = float(price_str)
                    if i == 1 and j == 1:
                        break_price = price_int
                        print(price_int, break_price)
                    print(abs((price_int - break_price) / break_price))
                    if abs((price_int - break_price) / break_price) > 0.1:
                        break
                    way = tr.find_all('td')[6].text.replace('\n', '').replace(' ', '')
                    print(goods_name, user_name, send_method, price_int, way)

    def get_part_goods_urls(self, all_goods_page):
        print(all_goods_page)
        goods_name_list = []
        goods_url_list = []
        try:
            all_goods_soup = self.get_soup(all_goods_page)
            all_goods_info = all_goods_soup.select(
                'body div[class="market-list"] div[class="l_Layout"] div[class="market-card"] '
                'div[class="list_card unhover"] ul[class="card_dota2"]')[0]

            # all_goods_info
            for li in all_goods_info.find_all('li'):
                goods_url = li.select('a')[0]['href']
                goods_name = li.select('a')[0]['title']
                goods_price=float(li.select('p strong[class="f_Strong"]')[0].text.replace('￥', ''))
                print(goods_name, goods_url, goods_price)
                goods_name_list.append(goods_name)
                goods_url_list.append([goods_url, goods_price])
            return pd.DataFrame(goods_url_list, index=goods_name_list, columns=['url', 'price'])
        except Exception as error:
            print(all_goods_page, error)
            return pd.DataFrame(goods_url_list, index=goods_name_list, columns=['url', 'price'])

    def get_goods_urls(self):
        a = time.time()
        all_goods_list=[]
        init_page = f'https://buff.163.com/market/?game=dota2#tab=selling&page_num=1'
        init_soup = self.get_soup(init_page)
        page_num = init_soup.select('body div[class="market-list"] div[class="l_Layout"] div[class="market-card"] '
                                    'div[class="pager card-pager light-theme simple-pagination"] '
                                    'ul')[0].find_all('li')[-2].text
        print(page_num)
        tpool = ThreadPool(10)
        for i in range(int(page_num)):
            all_goods_page = f'https://buff.163.com/market/?game=dota2#tab=selling&page_num={i+1}&sort_by=price.desc'
            # goods_name_url_list = self.get_part_goods_urls(all_goods_page)
            all_goods_list.append(tpool.apply_async(self.get_part_goods_urls, (all_goods_page,)))
            # self.get_part_goods_urls(all_goods_page)
            # print(all_goods_page)
            # all_goods_soup = self.get_soup(all_goods_page)
            # all_goods_info = all_goods_soup.select(
            #     'body div[class="market-list"] div[class="l_Layout"] div[class="market-card"] '
            #     'div[class="list_card unhover"] ul[class="card_dota2"]')[0]
            #
            # # all_goods_info
            # for li in all_goods_info.find_all('li'):
            #     goods_url = li.select('a')[0]['href']
            #     goods_name = li.select('a')[0]['title']
            #     print(goods_name, goods_url)
            #     goods_name_url_list.append([goods_name, goods_url])
                # goods_url_list.append(goods_url)
                # goods_name_list.append(goods_name)
        tpool.close()
        tpool.join()
        all_goods_df = pd.concat([x.get() for x in all_goods_list], axis=0, sort=True)
        b = time.time()
        print(f'cost time {b-a}')
        return all_goods_df

    def run(self):
        # all_goods_df = self.get_goods_urls()
        save_path = '/home/haishuowang/PycharmProjects/nl2sql_baseline-python3/all_goods_csv'
        all_goods_df = pd.read_csv('/home/haishuowang/PycharmProjects/nl2sql_baseline-python3/all_goods_df.scv'
                                   , sep='|', index_col=0)
        for x, y in all_goods_df.iloc[:100].iterrows():
            print(x, y['url'])
            self.get_goods_info(y['url'])


if __name__ == '__main__':
    # buff_spider = BuffSpider('', '')
    # all_goods_df = buff_spider.get_goods_urls()
    driver = webdriver.Chrome('/home/haishuowang/PycharmProjects/nl2sql_baseline-python3/data/chromedriver',
                              )
    driver.get('https://www.taobao.com/')
    # driver.get('https://list.tmall.com/search_product.htm?spm=a220o.7406545.1996500281.1.47da3779so8Byg&user_id=725677994&cat=51454011&active=1&style=g&acm=lb-zebra-27092-331834.1003.4.457096&sort=td&scm=1003.4.lb-zebra-27092-331834.OTHER_14434945515601_457096&industryCatId=51462017#J_Filter')
    # footer_login = driver.find_element_by_id("TANGRAM__PSP_3__footerULoginBtn")
    # username = driver.find_element_by_id("TANGRAM__PSP_3__userName")
    # password = driver.find_element_by_id("TANGRAM__PSP_3__password")
    # submit = driver.find_element_by_id("TANGRAM__PSP_3__submit")
    # print("**********************")
    # footer_login.click()
    # username.clear()
    # username.send_keys("1612255875@qq.com")
    # password.clear()
    # password.send_keys("qwer668993")
    # submit.click()
