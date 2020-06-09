import itchat
itchat.auto_login(hotReload=True)
import getpass, email, sys
from imapclient import IMAPClient
import ssl
import pandas as pd
import time
# friends = itchat.get_friends(update=True)[0:]
# for res in friends:
#     print(res['UserName'], res['PYQuanPin'], res['NickName'])


def run():
    try:
        email_name = 'WHS@yingpei.com'
        password = 'qwer668993'
        context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        connect = IMAPClient(host='mail.yingpei.com', ssl=True, ssl_context=context)
        try:
            connect.login(email_name, password)
        except connect.Error:
            print('Could not log in')

        connect.select_folder('Portfolio.Trade_Info', readonly=True)
        # connect.set_flags()
        raw_res = connect.search('ALL')
        user_name = None
        for i in itchat.get_chatrooms():
            print(i['NickName'], i['UserName'])
            if i['NickName'] == 'YP_FUT期货':
                user_name = i['UserName']
        if user_name is None:
            exit(-1)
        while True:
            new_res = connect.search('ALL')
            diff_res = list(set(new_res) - set(raw_res))
            if diff_res:
                print('_________________')
                print(diff_res)
                for res in diff_res:
                    msgdict = connect.fetch(res, ['BODY.PEEK[]'])
                    # connect.set_flags(res, b'\\SEEN', silent=False)
                    message = msgdict[res]
                    e = email.message_from_string(message[b'BODY[]'].decode('utf-8'))
                    exe_info = e['Subject'].split(']')[-1]
                    itchat.send(exe_info, user_name)
            raw_res = new_res
            time.sleep(2)
    except Exception as error:
        print(error)
        print(pd.datetime.now())
        time.sleep(2)
        run()


if __name__ == '__main__':
    run()
