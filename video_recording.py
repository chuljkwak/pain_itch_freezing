#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: chuljung kwak
"""

import os
import cv2
import time
from multiprocessing import Process
import pickle

def log(experiment_id, file_name,r):
    
    
    if 'time_stamp' not in file_name:
        
        print ('#{}'.format(r))
    
    
    file_name = 'result/' + file_name
    
    output = open (file_name+'.pkl','wb')
    pickle.dump(r,output)
    output.close()
    


def video_recording (experiment_id):
    
    try:
    
        cap = cv2.VideoCapture(0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        width,height = int(cap.get(3)),int(cap.get(4))
        
        out = cv2.VideoWriter('result/'+ experiment_id + '.avi', fourcc, fps, (width,height))
        
        r = []
        
        print ('video recording starts...')
        
        start_time = time.time()
        
        while time.time() - start_time < recording_time:
            
            print ('{} sec passed'.format(round(time.time() - start_time,1)), end = '\r', flush=True)
            
            ret, frame = cap.read()
            
            frame_time = time.time()
            
            if ret == True:
                
                #frame = cv2.resize(frame, (width,height), interpolation = cv2.INTER_AREA) 
                cv2.imshow('',frame)
                
                r.append(frame_time)
                
                out.write(frame)
                
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    
                    break
                
            else:
                
                break
            
        else:
            
            
            log(experiment_id, 'time_stamp_' + experiment_id,r)
            print ('recording is done')
            
            cap.release()
            out.release()
            
            
    
            
            
    except KeyboardInterrupt:
        
        log(experiment_id, 'time_stamp_' + experiment_id,r)
        
    cap.release()
    out.release()
    
    


if __name__ == '__main__':
    
    
    experiment_id = input ('experiment id: ')
    recording_time = input ('recording time in sec: ')
    recording_time = float(recording_time)
    
    files =  os.listdir('/home/pi/result')
    
       
    #P1 = Process(target = neuroom_utility.video_recording, args = (experiment_id,))
    P2 = Process (target = video_recording, args = (experiment_id,))
    #P3 = Process(target = neuroom_utility.wbs_training)

    #P1.start()            
    #time.sleep(3)
    P2.start()
    #P3.start()
    
    #P1.join()
    #P2.join()
    #P3.joib()
        
