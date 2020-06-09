from datetime import datetime
import requests
import urllib
# import urllib2
from bs4 import BeautifulSoup
import time
import os


link = 'https://auth.alipay.com/login/index.htm?loginScene=7&' \
       'goto=https%3A%2F%2Fauth.alipay.com%2Flogin%2Ftaobao_trust_login.' \
       'htm%3Ftarget%3Dhttps%253A%252F%252Flogin.taobao.com%252Fmember%252' \
       'Falipay_sign_dispatcher.jhtml%253Ftg%253Dhttps%25253A%25252F%25252F' \
       'www.taobao.com%25252F&params=VFBMX3JlZGlyZWN0X3VybD1odHRwcyUzQSUyRiUy' \
       'Rnd3dy50YW9iYW8uY29tJTJG'
