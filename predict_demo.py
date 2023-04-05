#!/usr/bin/env python
# encoding: utf-8
"""
File Description: prediction online
Author: nghuyong
Mail: nghuyong@163.com
Created Time: 2019-09-22 17:03
"""
import sys
sys.path.append('./bert/')
import bert.tokenization as tokenization
from bert.extract_features import InputExample, convert_examples_to_features
import numpy as np
import requests
import os
import time

vocab_file = os.environ.get('vocab_file', 'models/roberta_wwm_ext/vocab.txt')
max_token_len = os.environ.get('max_token_len', 128)

tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab_file, do_lower_case=True)

def preprocess(text):
    text_a = text
    example = InputExample(unique_id=None, text_a=text_a, text_b=None)
    feature = convert_examples_to_features([example], max_token_len, tokenizer)[0]
    input_ids = np.reshape([feature.input_ids], (1, max_token_len))
    return {
        "inputs": {"input_ids": input_ids.tolist()}
    }


if __name__ == '__main__':
    while True:
        # text = input("Input test sentence:\n")
        text = '爱德华纽盖特2的视频'
        start = time.time()
        #print(preprocess(text))
        txt = ["爱德华纽盖特2的视频",
               '怂包男友打脸名场面马思纯哭晕在厕所',
               '萌探探最新路透杨紫变红孩儿好可爱宋亚轩扮白龙马超帅气',
               'GDDony的视频',
               '這麼遠那麼競的视频',
               '美女时尚走秀这复古的穿着富有贵妇风范']
        tmp = []
        for a in txt:
            print(time.time())
            time.sleep(1)
            #text = x
            #text = {'inputs': {'input_ids': [[101, 138, 1677, 1677, 140, 4692, 1168, 1568, 172, 172, 172, 2340, 671, 1469, 1381, 671, 1469, 704, 7313, 4638, 3683, 6772, 2358, 172, 172, 138, 1506, 1506, 140, 138, 1506, 1506, 140, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}}
            #resp = requests.post('http://172.17.13.200:8503/v1/models/sogou:predict', json=preprocess(text))
            #resp = requests.post('http://172.17.13.200:8503/v1/models/cctv_news:predict', json=preprocess(text))
            #resp = requests.post('http://172.17.13.16:8503/v1/models/cctv_news:predict', json=text)
            #resp = requests.post('http://82.157.166.93:8503/v1/models/weibo_comments:predict', json=preprocess(text))
            resp = requests.post('http://172.17.13.110:8503/v1/models/sensitive_identify:predict', json=preprocess(a))
            #resp = requests.post('http://124.127.132.6:8503/v1/models/sensitive_identify:predict', json=preprocess(text))

            #labels = ['健康', '经济', '科教', '三农', '旅游', '体育', '新闻', '综艺']
            #labels = ['涉黄', '涉政', '正常']
            labels = ['0', '1']
            # labels = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
            # pro_0, pro_1 = resp.json()['outputs'][0]
            resp_array = np.array(resp.json()['outputs'][0])
            max_index = np.where(resp_array == np.max(resp_array))
            print(a)
            print(resp.json())
            print(resp_array[max_index[0][0]])
            print(f"label: {labels[max_index[0][0]]}")
            tmp.append(resp_array[max_index[0][0]])
        end = time.time()
        print(tmp)
        break
        '''
        #labels = ['时尚圈', '育儿', '科技咖', '搞笑', '汽车控', '养生堂', '旅游', '私房话', '职场', '星座', '生活家', '美食', '体育', '财经迷', '热门', '萌宠',
        #  '教育', '历史', '八卦精', '游戏', '军事']
        #labels = [ '健康', '经济', '科教', '三农', '旅游','体育', '新闻', '综艺' ]
        #labels = ['0', '1']
        #labels = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
        # pro_0, pro_1 = resp.json()['outputs'][0]
        resp_array = np.array(resp.json()['outputs'][0])
        max_index = np.where(resp_array == np.max(resp_array))
        print(resp.json())
        print(resp_array[max_index[0][0]])
        print(f"label: {labels[max_index[0][0]]} time consuming:{int((end - start) * 1000)}ms")
        '''