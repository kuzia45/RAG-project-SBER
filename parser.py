from bs4 import BeautifulSoup
import requests
import time
#import urllib2

import warnings
warnings.filterwarnings("ignore")

def find_files():
    url = "https://ai.gov.ru/knowledgebase/"
    TIMESLEEP = 60*10
    print('First request')
    req = requests.get(url, verify=False)
    while req.status_code == 503:
        print('Status code', req.status_code)
        time.sleep(TIMESLEEP)
        req = requests.get(url, verify=False)
    # print('Status code', req.status_code)
    soup = BeautifulSoup(req.text, features="html.parser")

    hrefs = []
    count_hrefs = 0
    print('Main Cycle')
    while count_hrefs < 200:
        hrefs_list = soup.find_all('a', class_ = 'knowledgeBaseCard__title')
        for a in hrefs_list:
            hrefs.append(a['href'])
            count_hrefs += 1
        time.sleep(1)
        if  len(hrefs_list) == 0:
            break
        req = requests.get(url, verify=False, params={'pageStart' : count_hrefs})
        while req.status_code == 503:
            print('Status code', req.status_code)
            time.sleep(TIMESLEEP)
            req = requests.get(url, verify=False, params={'pageStart' : count_hrefs})
        print('Number of hrefs counted', count_hrefs)
        soup = BeautifulSoup(req.text, features="html.parser")
    print('Finished finding hrefs')
    return hrefs

list_of_links = find_files()
print(len(list_of_links))
with open('hrefs.txt', 'w') as f:
    for href in list_of_links:
        f.write(href + '\n')
