import numpy as np
import cv2
import os
import os.path as osp
from util import *

LP_AIRBUS={}
LP_AIRBUS['Nose']=[SIZE_IMG//2,SIZE_IMG//5]
LP_AIRBUS['Fuselage']=[SIZE_IMG//2,SIZE_IMG//5*2]
LP_AIRBUS['Empennage']=[SIZE_IMG//2,SIZE_IMG//5*4]
LP_AIRBUS['FLwing']=[SIZE_IMG//5,SIZE_IMG//2]
LP_AIRBUS['FRwing']=[SIZE_IMG//5*4,SIZE_IMG//2]
LP_AIRBUS['BLwing']=[SIZE_IMG//5*2,SIZE_IMG//5*4]
LP_AIRBUS['BRwing']=[SIZE_IMG//5*3,SIZE_IMG//5*4]


def draw_airbus():
    LENTH_HALF=SIZE_IMG//2-LP_AIRBUS['Nose'][1]
    WIDTH_HALF=SIZE_IMG//25
    WIDTH_BACKWING=SIZE_IMG//25
    img = np.zeros((SIZE_IMG,SIZE_IMG,3), np.uint8)
    pts = np.array([LP_AIRBUS['Fuselage'],LP_AIRBUS['FLwing'],LP_AIRBUS['FRwing']], np.int32)
    img = cv2.fillPoly(img,[pts],(0,255,255))
    img =cv2.rectangle(img,tuple(np.array(LP_AIRBUS['BLwing'])-np.array([0,WIDTH_BACKWING])),tuple(LP_AIRBUS['BRwing']),(0,0,255),-1)
    img =cv2.ellipse(img,(SIZE_IMG//2,SIZE_IMG//2),(LENTH_HALF,WIDTH_HALF),90,0,360,(255,0,0),-1)
    return img

LP_FIGHTER={}
LP_FIGHTER['Nose']=[SIZE_IMG//2,SIZE_IMG//5]
LP_FIGHTER['Fuselage']=[SIZE_IMG//2,SIZE_IMG//5*2]
LP_FIGHTER['Empennage']=[SIZE_IMG//2,SIZE_IMG//5*4]
LP_FIGHTER['FLwing']=[SIZE_IMG//3,SIZE_IMG*2//3]
LP_FIGHTER['FRwing']=[SIZE_IMG*2//3,SIZE_IMG*2//3]
LP_FIGHTER['BLwing']=[SIZE_IMG//5*2,SIZE_IMG//5*4]
LP_FIGHTER['BRwing']=[SIZE_IMG//5*3,SIZE_IMG//5*4]

def draw_fighter():
    LENTH_HALF=SIZE_IMG//2-LP_FIGHTER['Nose'][1]
    WIDTH_HALF=SIZE_IMG//50
    TOP_BACKWING=SIZE_IMG*7//10
    TOP_BACKWING_x = (LP_FIGHTER['BLwing'][0]+LP_FIGHTER['BRwing'][0])//2
    img = np.zeros((SIZE_IMG,SIZE_IMG,3), np.uint8)
    pts = np.array([LP_FIGHTER['Fuselage'],LP_FIGHTER['FLwing'],LP_FIGHTER['FRwing']], np.int32)
    img = cv2.fillPoly(img,[pts],(0,255,255))
    pts = np.array([[TOP_BACKWING_x, TOP_BACKWING],LP_FIGHTER['BLwing'],LP_FIGHTER['BRwing']], np.int32)
    img = cv2.fillPoly(img,[pts],(0,0,255))
    img =cv2.ellipse(img,(SIZE_IMG//2,SIZE_IMG//2),(LENTH_HALF,WIDTH_HALF),90,0,360,(255,0,0),-1)
    return img

LP_UAV={}
LP_UAV['Nose']=[SIZE_IMG//2,SIZE_IMG//3]
LP_UAV['Fuselage']=[SIZE_IMG//2,SIZE_IMG//2]
LP_UAV['Empennage']=[SIZE_IMG//2,SIZE_IMG*2//3]
LP_UAV['FLwing']=[SIZE_IMG*4//25,SIZE_IMG//2]
LP_UAV['FRwing']=[SIZE_IMG*21//25,SIZE_IMG//2]
LP_UAV['BLwing']=[SIZE_IMG//5*2,SIZE_IMG*2//3]
LP_UAV['BRwing']=[SIZE_IMG//5*3,SIZE_IMG*2//3]
def draw_uav():
    LENTH_HALF=SIZE_IMG//2-LP_UAV['Nose'][1]
    WIDTH_HALF=SIZE_IMG//50
    WIDTH_BACKWING=SIZE_IMG//25
    WIDTH_WING=SIZE_IMG//25
    img = np.zeros((SIZE_IMG,SIZE_IMG,3), np.uint8)
    img =cv2.rectangle(img,tuple(np.array(LP_UAV['FLwing'])-np.array([0,WIDTH_WING])),tuple(LP_UAV['FRwing']),(0,255,255),-1)
    img =cv2.rectangle(img,tuple(np.array(LP_UAV['BLwing'])-np.array([0,WIDTH_BACKWING])),tuple(LP_UAV['BRwing']),(0,0,255),-1)
    img =cv2.ellipse(img,(SIZE_IMG//2,SIZE_IMG//2),(LENTH_HALF,WIDTH_HALF),90,0,360,(255,0,0),-1)
    return img

import random 
def make_data(PATH_DATA_SET,name_data,num):
    PATH_DATA=osp.join(PATH_DATA_SET,name_data)
    if not osp.exists(PATH_DATA):
        os.makedirs(PATH_DATA)
    if name_data=='AIRBUS':
        img=draw_airbus()
        p_sets= LP_AIRBUS    
    elif name_data=='FIGHTER':
        img=draw_fighter()
        p_sets= LP_FIGHTER
    elif name_data=='UAV':
        img=draw_uav()
        p_sets= LP_UAV
    else:
        print('wrong data')
    
    for i in range(num):
        angle=random.randint(0, 359)
        anno_file = "{}_{}.txt".format(name_data, angle)
        anno_file=osp.join(PATH_DATA,anno_file)
        img_file = "{}_{}.jpg".format(name_data, angle)
        img_file=osp.join(PATH_DATA,img_file)
        rotate_anno(anno_file,angle,p_sets)
        matRotate = cv2.getRotationMatrix2D((SIZE_IMG*0.5, SIZE_IMG*0.5), angle, 1) # mat rotate 1 center 2 angle 3 缩放系数
        img_r = cv2.warpAffine(img, matRotate, (SIZE_IMG, SIZE_IMG))
        cv2.imwrite(img_file,img_r)

if __name__ == '__main__':

    PATH_DATA_SET=os.path.join('F:','AIRPLANES')
    PATH_SHOW_SAVE=os.path.join('F:','AIRPLANES_SHOW')
    make_data(PATH_DATA_SET,'AIRBUS',10)
    make_data(PATH_DATA_SET,'FIGHTER',10)
    make_data(PATH_DATA_SET,'UAV',10)
    
    draw_points(PATH_DATA_SET,PATH_SHOW_SAVE)


