# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 19:49:55 2019

@author: SAM

category based accuracy
"""

intervals = [12,21,24,12,16,12,13,14,15,15,13,10,24,16,31,22,12, 8,10,10,13,15,23,18,10, 7,18,26,21,22,15]

result = [True,True,True,True,True,True,True,True,True,False,True,True,False,False,False,False,False,True,False,True,False,False,True,False,False,True,True,True,False,True,True,False,True,True,True,False,False,False,False,False,True,False,False,False,True,True,False,False,True,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,True,False,True,True,True,True,False,False,False,False,False,False,False,False,False,False,False,True,True,False,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,False,False,True,False,True,True,False,False,False,True,False,False,False,True,False,False,True,True,False,False,True,True,True,True,True,False,False,True,False,False,False,True,True,False,True,False,False,True,False,True,True,False,False,True,True,True,True,True,False,False,False,False,False,False,False,False,False,False,True,True,True,True,True,False,False,False,False,False,False,False,False,False,True,True,False,True,True,True,True,True,True,False,True,True,True,False,False,False,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,True,True,True,False,True,True,False,True,True,False,True,True,False,True,True,False,False,False,False,False,True,False,True,True,True,False,False,False,False,False,True,False,True,True,False,True,False,False,False,True,True,True,True,True,True,True,False,False,True,True,True,False,False,True,False,True,True,True,False,False,False,True,False,True,False,True,False,False,False,False,False,True,False,False,False,False,False,False,False,True,True,False,False,False,True,True,True,False,False,False,False,False,False,False,False,False,False,True,True,False,False,False,False,False,True,True,True,True,False,True,False,True,False,True,False,True,True,True,False,False,False,True,False,False,False,True,False,False,True,True,False,True,True,True,True,True,False,False,False,True,True,True,False,True,True,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,True,False,True,True,True,True,True,True,True,True,False,False,True,True,False,True,True,False,True,True,True,False,True,True,True,False,False,False,True,True,True,True,True,True,True,True,True,True,True,False,False,False,False,True,True,True,True,True,True,True,True,True,False,True,False,True,True,False,True,True,True,False,True,True,True,True,False,True,True,False,True,True,True,True,True,True,True,True,True,True,True,False,False,True,True,True,True,True,False,True,True,True,True,True,False,True,True,True,True,True,True,True,True,True]
classes= ['back_pack','bike','bike_helmet','bookcase','bottle','calculator','desk_chair','desk_lamp','desktop_computer','file_cabinet','headphones','keyboard','laptop_computer','letter_tray','mobile_phone','monitor','mouse','mug','paper_notebook','pen','phone','printer','projector','punchers','ring_binder','ruler','scissors','speaker','stapler','tape_dispenser','trash_can']
count = 0
scores = {}
for index , interval in enumerate(intervals):
    category = result[count : count+ interval]
    bad_predictions =  sum(category)
    score = bad_predictions / interval
    scores[classes[index]] = 1- score
    count = count + interval
    
    
from collections import Counter


Counter(scores).most_common()

    