# -*- coding: utf-8 -*-
"""
@author: chuljung kwak
"""

from skimage.metrics import structural_similarity as ssim
import os
import cv2
import numpy as np
import pickle
import pandas as pd
import glob

def read_pkl(filename):
    
    x = pd.read_pickle(filename)
    
    return x


def save_pkl(data,filename):   
    
    output = open(filename, 'wb')
    pickle.dump(data, output)
    output.close()    
    
def slice_list_by_ascending (alist,size):
    #slice list like [1,2,3,4,5] ->[[1,2,3],[2,3,4],[3,4,5]]
    
    
    r = []
    
    for i,j in enumerate(alist[:-size+1]):
        
        r.append(alist[i:i+size])
        
    return r

def slice_list_when_this_much_diff (alist,diff_size = 0.03):

    r = [[alist[0]]]

    for i in enumerate(alist[:-1]):

        if abs(alist[i[0]] - alist[i[0]+1]) < 0.1:

            r[-1].append(alist[i[0]+1])

        else:

            r.append([alist[i[0]+1]])

    return r

def folder_file_name_from_path (path):
    
    if '/' in path:
        
        idx = path[::-1].index('/')
        
    elif '\\' in path:
        
        idx = path[::-1].index('\\')
    
    return path[:-idx], path[-idx:]

def find_files_with_keyword (folder, keyward, rec=False):
    
    r = glob.glob(folder + '/**', recursive=rec)
    
    r = [_ for _ in r if keyward in _]
    
    return r

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def compare_two_frames (file_name):
    
    cap = cv2.VideoCapture(file_name)
    
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    diffs = [0]
    
    for frame_no in range(video_length-1):
        
        perc = round(frame_no/video_length*100,1)
        
        print('\rprocess completed {} %'.format(str(perc)), end='')
        
        cap.set(1, frame_no)
        ret, firstframe = cap.read()
        cap.set(1, frame_no+1)
        ret, secondframe = cap.read()
        
        #diff = ssim(firstframe, secondframe, channel_axis=-1, data_range=255)
        diff = mse(firstframe, secondframe)
        
        diffs.append(diff)    
        
    save_as = file_name[:-4] + '_frame_diffs.pkl'
    save_pkl(diffs, save_as)
    
    
def analysis (filename, time_bins = [[100,200],[200,300]]):
    
    time_window = 10
    freezing_threshold = 20
    
    cap = cv2.VideoCapture(filename)
    
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    diff_file = filename[:-4] + '_frame_diffs.pkl'
    
    r = []
    
    if os.path.isfile(diff_file):
        diff_file = read_pkl(diff_file)
        
    else:
        print ('frame differences have not been analyzed...')
        print ('so, the differences btw frames are compared first...')
        compare_two_frames(filename)
        diff_file = read_pkl(diff_file)
        
    time_stamps_infolder = folder_file_name_from_path(filename)
    time_stamps_infolder = find_files_with_keyword(time_stamps_infolder[0], 'time_stamp')
    time_stamps_infolder = [_ for _ in time_stamps_infolder if '.pkl' in _]
    
    time_stamp = []
    
    for time_stamp_infolder in time_stamps_infolder:
        
        if len(read_pkl(time_stamp_infolder)) == video_length:
            
            time_stamp.append(read_pkl(time_stamp_infolder))
            
            
    print ('adding freezing data of {}'.format(filename))
    
    if len (time_stamp) == 1:
        
        time_stamp = time_stamp[0]
        time_stamp = [_ - time_stamp[0] for _ in time_stamp]
        
        for time_bin in time_bins:
            
            time_index = [idx for idx,vlu in enumerate(time_stamp) if time_bin[0] < vlu < time_bin[1]]
            
            diff_in_time_bin = slice_list_by_ascending(diff_file[time_index[0] - time_window + 1: time_index[-1]+1],time_window)
            
            time_stamp_in_time_bin = slice_list_by_ascending(time_stamp[time_index[0] - 9: time_index[-1]+1],10)
            
            fr_in_time_bin =  [tt[-1]  for tt,dd in zip(time_stamp_in_time_bin,diff_in_time_bin) if np.mean(dd) < freezing_threshold]
            fr_in_time_bin = slice_list_when_this_much_diff(fr_in_time_bin)
            fr_in_time_bin = [_ for _ in fr_in_time_bin if _[-1] - _[0] > 0.5]
            fr_in_time_bin = sum(fr_in_time_bin,[])
            
            
            #diff_in_time_bin = ['f' if np.mean(_) < 20 else 'nf' for _ in diff_in_time_bin]
            
            freezing_perc = len(fr_in_time_bin)/len(diff_in_time_bin)*100
            
            print (freezing_perc)
            
            r.append(freezing_perc)
            
        return r
            
            
        
    else:
        
        print ('there are no matching time stamp for the video file')
        
        
def folder_run (foldername, time_bins = [[100,200],[200,300]], 
                extensions = ['avi','mp4']):
    
    r = {}
    
    vfiles = find_files_with_keyword(foldername, '')
    vfiles = [_ for _ in vfiles if _[-3:] in extensions]
    
    for vfile in vfiles:
        
        r[folder_file_name_from_path(vfile)[1]] = analysis(vfile, time_bins)
        
    csv_save_as = '{}/results.csv'.format(foldername)
        
    pd.DataFrame.from_dict(data=r, orient='index').to_csv(csv_save_as, header=False)
        
    return r
    
    
    
    
def create_freezing_video(file_name,
                          time_window = 10,freezing_threshold = 20):
    
    
    
    cap = cv2.VideoCapture(file_name)
    
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_diffs = file_name[:-4] + '_frame_diffs.pkl'
    frame_diffs = [0] + read_pkl(frame_diffs)
    
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  
    fps = cap.get(cv2.CAP_PROP_FPS)

    
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    save_as = file_name[:-4] + '_freezing.avi'
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_as, fourcc, fps, (int(width),int(height)))
    
    if (cap.isOpened()== False): 
      print("Error opening video stream or file")
      
    frame_no = 0
     
    while(cap.isOpened()):
      ret, frame = cap.read()
      if ret == True:
        
          
        perc = round(frame_no/video_length*100,1)
        
        print('\rprocess completed {} %'.format(str(perc)), end='')
        
        #cv2.putText(frame,"Hello World!!!", (10,10), cv2.CV_FONT_HERSHEY_SIMPLEX, 2, 255)
        
        first_frame_no = frame_no - time_window if frame_no - time_window >= 0 else 0
        last_frame_no = frame_no + time_window if frame_no + time_window <= video_length else video_length
        
        #if frame_diffs[frame_no] < 50:
        if np.mean(frame_diffs[first_frame_no:frame_no]) < freezing_threshold:
            
            #cv2.putText(frame, str(frame_diffs[frame_no])[:4], (10,30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Freezing!!!', (10,30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Freezing',frame)
        
        frame_no += 1
        
        out.write(frame)
        
     
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break
      else: 
        break
     
    cap.release()
    out.release()
    cv2.destroyAllWindows()
        
