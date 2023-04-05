import os
import pandas as pd
from pandas.core.frame import DataFrame

content = []

for file in os.listdir('orginal'):
    src_file = os.path.join(os.getcwd(), 'orginal', file)
    label = file[:-4]

    with open(src_file, encoding='utf-8') as f:
        print(src_file)
        one_sample = ''
        for line in f.readlines():
            newStr = line.replace(" ", "").replace("\t", "").strip('\n').strip('\r').replace("\"",'')
            label = newStr[-1:]
            con = newStr[:-1]
            if label not in ['0', '1', '2'] or not con:
                print('con' + con + 'newStr' + newStr)
                print(label)
                continue

            if label == '0':
                label = '涉黄'
            elif label == '1':
                label = '涉政'
            else:
                label = '正常'

            content.append(label + '\t' + con)
    
    print(len(content))        

    with open(file, 'a', encoding='utf-8') as fp:
        for each in content:
            fp.write(each)
            fp.write('\n')

    content.clear()



        
       