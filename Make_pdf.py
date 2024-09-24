from bs4 import BeautifulSoup
import requests
import time
import warnings
warnings.filterwarnings("ignore")

def wait_response(func):
    def wrapper(url):
        TIMESLEEP = 60*8
        TIMESLEEPSMALL = 1
        time.sleep(TIMESLEEPSMALL)
        status_code, req = func(url)
        while status_code != 200:
            #DEBUG
            print('Wait response, status code = ', status_code)
            time.sleep(TIMESLEEP)
            status_code, req = func(url)
        return status_code, req
    return wrapper


@wait_response
def get_request(url):
    # print('!!!!!!')
    # print(url)
    req = requests.get(url, verify=False)
    return req.status_code, req


def get_hrefs(filename):
    hrefs = []
    with open(filename, 'r') as f:
        hrefs = [line.rstrip('\n') for line in f.readlines()]
    return(hrefs)


def make_full_href(path, href):
    return(path + href)


def getsave_pdf(href, dir_to_save):
    _, response = get_request(href)
    soup = BeautifulSoup(response.content, 'html.parser')
    pdf = soup.select_one('.docListCard')
    #pdf_toload.append('https://ai.gov.ru' + pdf['href'])
    if pdf: 
        _, response_pdf = get_request('https://ai.gov.ru' + pdf['href'])
        download_pdf(response_pdf, dir_to_save, href)


def download_pdf(response, dir_to_save, href):
    print(dir_to_save + f"{href.split('/')[-2]}.pdf")
    with open(dir_to_save + str(hash(f"{href.split('/')[-2]}")) + ".pdf", 'wb') as file:
        file.write(response.content)

if __name__ == "__main__":
    href_file = 'hrefs.txt'
    dir_to_save = './files/'
    path_http = "https://ai.gov.ru"
    hrefs = get_hrefs(href_file)
    for h in hrefs:
        full_h = make_full_href(path_http, h)
        getsave_pdf(full_h, dir_to_save)
        #download_pdf(full_href, dir_to_save)