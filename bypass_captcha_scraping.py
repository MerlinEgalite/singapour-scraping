# import web driver
from selenium import webdriver
from parsel import Selector
import urllib
import os
import sched
import time
from selenium.webdriver.common.keys import Keys
from solve_captcha import *


OUTPUT_FOLDER = 'real_captcha_dataset'
list_of_request = ['singapour']

# specifies the options to the chromedriver.exe
options = webdriver.ChromeOptions()
#options.add_argument('--headless')


url = 'https://www.tis.bizfile.gov.sg'


driver = webdriver.Chrome('/Users/merlinegalite/Desktop/octobot/Scraping/LinkedInScraping/chromedriver', options=options)


driver.get('https://www.bizfile.gov.sg/ngbbizfileinternet/faces/oracle/webcenter/portalapp/pages/BizfileHomepage.jspx?_afrWindowId=null&_afrLoop=11499874782621942&_afrWindowMode=0&_adf.ctrl-state=10irrn140w_4#%40%3F_afrWindowId%3Dnull%26_afrLoop%3D11499874782621942%26_afrWindowMode%3D0%26_adf.ctrl-state%3D2w324sfb3_4')

query_button = driver.find_element_by_xpath('//*[@class="search_Icon2 af_commandImageLink p_AFTextOnly"]')
query_button.click()

time.sleep(4)


sel = Selector(text=driver.page_source)


img = sel.xpath('//*[@id="pt1:r1:0:r1:0:i1"]')
src = img.attrib['src']
print(url + src)


save_path = os.path.join(OUTPUT_FOLDER, "newcaptcha_to_bypass.png")
urllib.request.urlretrieve(url + src, save_path)
