from genericpath import isdir
import pandas as pd
import os

from selenium import webdriver


def all_compositions(update=True, persist=False):
    all_indexes = get_all_indexes(update=update)

    stocks = set()
    for index in all_indexes.keys():
        if all_indexes[index] is not None:
            for ticker in all_indexes[index]['ticker'].values:
                stocks.add(ticker)

    joined_df = pd.DataFrame(columns= ['company']+list(all_indexes.keys()), index=stocks )

    for index in all_indexes.keys():
        if all_indexes[index] is not None:
            for i in range(all_indexes[index].shape[0]):
                joined_df['company'].loc[ all_indexes[index]['ticker'].iloc[i] ] = all_indexes[index]['stock'].iloc[i]
                joined_df[index].loc[ all_indexes[index]['ticker'].iloc[i] ] = all_indexes[index]['composition'].iloc[i]
    if persist:
        joined_df.to_csv('./data/index/compositions.csv')
    
    return joined_df

def get_all_indexes(update=True):
    '''

    '''
    indexes = ['IBOV','IBXX','IBRA','IGCX','ITAG','IGNM','IGCT','IDIV','MLCX','SMLL','IVBX','ICO2','ISEE','ICON','IEEX','IFNC','IMOB','INDX','IMAT','UTIL','IFIX','BDRX']

    compositions = {index : get_benchmark(benchmark=index, update=update) for index in indexes}
    return compositions         
    

def get_benchmark(benchmark='IBOV', update=True):
    '''
        benchmark: {'IBOV', 'SMLL', 'IBRA', 'IBXX'} # 'IBOV' -> Ibovespa, 'SMLL' -> Small Caps, 'IBRA' -> IBRA, 'IBXX' -> IBrX100
        update: boolean # if True, then fetch composition in B3's website, else load the persisted data
    
    '''
    benchmark=benchmark.upper()
    
    path = os.path.join(os.path.relpath('.'),'data', 'index')
    make_dir(path)
    path = os.path.realpath(path)

    if update:
        df = download_benchmark(benchmark, path=path)
    else:
        df = get_file(path, text=benchmark)
    return df

def download_benchmark(benchmark, delete_previous=True, path=os.getcwd()):
    if delete_previous:
        delete_previous_files(benchmark, path)   

    driver = set_driver(url=f'https://sistemaswebb3-listados.b3.com.br/indexPage/day/{benchmark}?language=pt-br', path=path)
    driver.find_element_by_link_text('Download').click()
    driver.quit()

    df = get_file(path, text=benchmark)
    return df

def get_file(path=None, text='IBOV'):
    if path is None:
        path = os.path.join(os.getcwd(),'data', text)
    df = None
    for file in os.listdir(path):
        if text in file:
            df = pd.read_csv(path+'/'+file, sep=';', thousands='.', decimal=',', skiprows=1, index_col=False)
            df = df_setup(df)
    return df

def df_setup(df):
    df = df.iloc[0:df.shape[0]-2]
    df = df.rename(columns={'C�digo':'ticker', 'A��o':'stock', 'Tipo':'type', 'Qtde. Te�rica':'stock_number', 'Part. (%)':'composition'})    
    df['composition'] /= 100
    return df

def delete_previous_files(text, path):
    for file in os.listdir(path):
        if text in file:
            os.remove(path+'/'+file)


def set_driver(url, path=os.getcwd() ):       
    files_types = 'application/zip,application/octet-stream,application/x-zip-compressed,multipart/x-zip,application/x-rar-compressed, application/octet-stream,application/msword,application/vnd.ms-word.document.macroEnabled.12,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/rtf,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel,application/vnd.ms-word.document.macroEnabled.12,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/xls,application/msword,text/csv,application/vnd.ms-excel.sheet.binary.macroEnabled.12,text/plain,text/csv/xls/xlsb,application/csv,application/download,application/vnd.openxmlformats-officedocument.presentationml.presentation,application/octet-stream'
    fireFoxOptions = webdriver.FirefoxOptions()
    fireFoxOptions.set_headless()

    profile = webdriver.FirefoxProfile()
    profile.set_preference('browser.download.folderList', 2)
    profile.set_preference('browser.download.manager.showWhenStarting', False)
    
    profile.set_preference('browser.download.dir', path)
    profile.set_preference('browser.helperApps.neverAsk.saveToDisk', files_types)
    profile.set_preference('general.warnOnAboutConfig', False)
    profile.update_preferences()
    

    driver = webdriver.Firefox(executable_path='./webdriver/geckodriver', firefox_options=fireFoxOptions, firefox_profile=profile)
    driver.get(url)
    return driver

def make_dir(path='./'):
    if os.path.isdir(path):
        return
    list_to_string = lambda seq, sep=',' : sep.join(str(i) for i in seq)
    dir_list = [d for d in path.split('/') if len(d) > 0]

    for i in range(1, len(dir_list)+1):
        directory = list_to_string(dir_list[0:i], sep='/')
        if not os.path.isdir(directory):
            os.mkdir(directory)
