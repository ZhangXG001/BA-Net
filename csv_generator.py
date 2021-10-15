#coding:utf-8
import os
import csv    
def create_csv(dirname):    
    path = './dataset/'+ dirname +'/'    
    name = os.listdir(path)  
    name.sort(key=lambda x: (x.split('_')[0][-1:]))
    #print(name)
    with open (dirname+'.csv','w', newline='') as csvfile:    

        writer = csv.writer(csvfile)  
        for n in name:
            if n[-4:] == '.jpg':                 
                writer.writerow(['./dataset/'+str(dirname) +'/'+ str(n),'./dataset/' + str(dirname) + 'label/' + str(n[:-4] + '.png')])    # 写入包含路径的原文件名和加PNG的文件名，方便model中读取文件
            else:
                pass

if __name__ == "__main__":  
    create_csv('train')
    create_csv('validation')
