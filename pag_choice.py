import time
import pyautogui as pag
screenWidth, screenHeight = pag.size()
pag.PAUSE = 1.5

# 下一页按键
next_page = 'next_page'
# 导出按键
button = 'button'
# 保存按键
save_button = 'save'
# 保存文件名
save_file = 'save_file'
# 关闭excel文件
kill_excel = 'kill_excel'
# 到最后一页的标志
# last_page = pag.locateOnScreen(f'/home/haishuowang/spider_data/last_page.png')
# ind_loc = pag.center(pag.locateOnScreen(f'/home/haishuowang/spider_data/{ind}.png'))
# button_loc = pag.center(pag.locateOnScreen(f'/home/haishuowang/spider_data/{button}.png'))
# next_page_loc = pag.center(pag.locateOnScreen(f'/home/haishuowang/spider_data/{next_page}.png'))
# save_button_loc = pag.center(pag.locateOnScreen(f'/home/haishuowang/spider_data/{save_button}.png'))
# save_file_loc = pag.center(pag.locateOnScreen(f'/home/haishuowang/spider_data/{save_file}.png'))
# kill_excel_loc = pag.center(pag.locateOnScreen(f'/home/haishuowang/spider_data/{kill_excel}.png'))


class PagChoice:
    @staticmethod
    def get_data(page_num=1):
        # 导出键
        button_loc = (3368, 631)
        # 下一页
        next_page_loc = (5227, 630)
        # 保存健
        save_button_loc = (4588, 1535)
        # save_file_loc = (3749, 1465)
        # 保存文件名
        save_click_loc = (4588, 1465)
        # 关闭excel
        kill_excel_loc = (4605, 1077)
        #

        while True:
            last_page = pag.locateOnScreen(f'/home/haishuowang/spider_data/last_page.png')
            print(page_num)
            pag.click(button_loc)
            pag.click(save_click_loc)
            pag.typewrite(str(page_num), interval=0.25)
            # pag.click(next_page_loc)
            print('save_button_click')
            pag.click(save_button_loc)
            time.sleep(4)
            pag.click(kill_excel_loc)
            if not last_page:
                pag.click(next_page_loc)
                page_num += 1
                time.sleep(1)

            else:
                print('last page')
                break

    def run(self):
        # ind_list = ['甲醇', '铁矿石', 'PTA', '豆粕', '动力煤', '焦炭', '焦煤', '石油沥青', '铜', '镍']
        ind_list = ['甲醇']
        for ind in ind_list:
            print(f'当前行业 {ind}')
            ind_loc = pag.center(pag.locateOnScreen(f'/home/haishuowang/spider_data/{ind}.png'))
            pag.click(ind_loc)
            time.sleep(2)
            self.get_data()


if __name__ == '__main__':
    PagChoice().run()
    # PagChoice().get_data(page_num=178)
