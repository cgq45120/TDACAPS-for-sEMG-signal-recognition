import scipy.io
import os
import numpy as np
import math
class ImportMatData():
    def __init__(self):
        self.length_windows = 300
        self.number_data = 195
        self.interval = 50
        self.channal = 16

    def load_data(self,data_class):
        if data_class == "person":
            self.path = '../all_data_wrist/'
            train_orig = np.zeros((self.number_data*15,self.length_windows,self.channal))
            train_flag = np.zeros((self.number_data*15,1))
            test_orig = np.zeros((self.number_data*5,self.length_windows,self.channal))
            test_flag = np.zeros((self.number_data*5,1))
            m_train = 0
            m_test = 0
            for files in os.listdir(self.path):
                EMG = scipy.io.loadmat(self.path + files)
                data = EMG['data'][:,:self.channal]
                if int(files[10]) <= 4:
                    for i in range(self.number_data):
                        train_orig[m_train * self.number_data + i, :, :] = data[i*self.interval:i*self.interval+self.length_windows,:]
                    train_flag[m_train * self.number_data : ( m_train + 1 ) * self.number_data] = np.tile(int(files[6]),(self.number_data,1))-1
                    m_train += 1
                elif int(files[10]) <= 6:
                    for i in range(self.number_data):
                        test_orig[m_test * self.number_data + i, :, :] = data[i*self.interval:i*self.interval+self.length_windows,:]
                    test_flag[m_test * self.number_data : ( m_test + 1 ) * self.number_data] = np.tile(int(files[6]),(self.number_data,1))-1
                    m_test += 1
            train_flag = train_flag.astype('int')
            test_flag = test_flag.astype('int')
            return train_orig,train_flag,test_orig,test_flag
        elif data_class == "people":
            self.path = '../all_data_wrist_people/'
            train_orig = np.zeros((self.number_data*210,self.length_windows,self.channal))
            train_flag = np.zeros((self.number_data*210,1))
            test_orig = np.zeros((self.number_data*30,self.length_windows,self.channal))
            test_flag = np.zeros((self.number_data*30,1))
            m_train = 0
            m_test = 0
            for files in os.listdir(self.path):
                EMG = scipy.io.loadmat(self.path + files)
                data = EMG['data'][:,:self.channal]
                if int(files[2]) <= 7:
                    for i in range(self.number_data):
                        train_orig[m_train * self.number_data + i, :, :] = data[i*self.interval:i*self.interval+self.length_windows,:]
                    train_flag[m_train * self.number_data : ( m_train + 1 ) * self.number_data] = np.tile(int(files[6]),(self.number_data,1))-1
                    m_train += 1
                elif int(files[2]) <= 8:
                    for i in range(self.number_data):
                        test_orig[m_test * self.number_data + i, :, :] = data[i*self.interval:i*self.interval+self.length_windows,:]
                    test_flag[m_test * self.number_data : ( m_test + 1 ) * self.number_data] = np.tile(int(files[6]),(self.number_data,1))-1
                    m_test += 1
            train_flag = train_flag.astype('int')
            test_flag = test_flag.astype('int')
            return train_orig,train_flag,test_orig,test_flag
class DealSign(object):
    def __init__(self):
        self.length_windows = 300
        self.number_data = 195
        self.channal = 16
        self.feature = 14

    def sumRMS(self,signalOrigin):
        sumRms = np.sqrt(np.sum(signalOrigin**2/self.length_windows,axis=0))
        return sumRms

    def sumMAV(self,signalOrigin):
        sumMAV = np.sum(abs(signalOrigin),axis=0)
        return sumMAV

    def sumWL(self,signalOrigin):
        sumWL = np.sum(abs(signalOrigin[0:299,:] - signalOrigin[1:,:])/299,axis=0)
        return sumWL

    def sumZC(self,signalOrigin):
        condition1 = (abs(signalOrigin[0:299,:]-signalOrigin[1:])>50)+0
        condition2 = (np.multiply(signalOrigin[0:299,:],signalOrigin[1:,:])<0)+0
        sumZC = np.sum(((condition1+condition2)>1)+0,axis=0)
        return sumZC

    def sumDASDV(self,signalOrigin):
        sumDASDV = np.sqrt(np.sum((signalOrigin[0:299,:] - signalOrigin[1:,:])**2/299,axis=0))
        return sumDASDV

    def sumLOG(self,signalOrigin):
        sumLOG = np.exp(np.sum(signalOrigin/self.length_windows,axis=0))
        return sumLOG

    def sumSII(self,signalOrigin):
        sumSII = np.sum(signalOrigin**2,axis=0)
        return sumSII

    def sumTM3(self,signalOrigin):
        sumTM3 = abs(np.sum(signalOrigin**3/self.length_windows,axis=0))
        return sumTM3

    def sumTM4(self,signalOrigin):
        sumTM4 = abs(np.sum(signalOrigin**4/self.length_windows,axis=0))
        return sumTM4

    def sumTM5(self,signalOrigin):
        sumTM5 = abs(np.sum(signalOrigin**5/self.length_windows,axis=0))
        return sumTM5

    def frequencyRatio(self,signalOrigin):
        sign = abs(np.fft.fft(signalOrigin, 512,axis = 0) / self.length_windows)
        sumfrequencyRatio = np.min(sign, axis = 0)/np.max(sign, axis = 0)
        return sumfrequencyRatio

    def sumIEMG(self,signalOrigin):
        sumIEMG = np.sum(abs(signalOrigin)/self.length_windows,axis=0)
        return sumIEMG

    def sumMFMN(self,signalOrigin):
        f = 1000*np.linspace(0,1,512)
        sign = abs(np.fft.fft(signalOrigin, 512,axis = 0) / self.length_windows)
        sumMFMN = np.sum(np.multiply(np.tile(f,(self.channal,1)).T,sign),axis=0)/np.sum(sign,axis=0)
        return sumMFMN

    def sumMFMD(self,signalOrigin):
        sign = abs(np.fft.fft(signalOrigin, 512,axis = 0) / self.length_windows)
        sumMFMD = np.sum(sign, axis = 0)/2
        return sumMFMD

    def readFile(self,data_class):
        data_import = ImportMatData()
        traindata,trainFlag,testdata,testFlag = data_import.load_data(data_class)
        trainData = self.deal(traindata)
        testData = self.deal(testdata)
        trainData,testData = self.onehot(trainData,testData,data_class)
        return trainData,trainFlag,testData,testFlag

    def onehot(self,trainData,testData,data_class):
        one_hot = np.vstack((trainData, testData))
        one_hot = (one_hot - one_hot.min(0))/(one_hot.max(0) - one_hot.min(0))
        if data_class == "person":
            trainData = one_hot[0:self.number_data*15, :]
            testData = one_hot[self.number_data*15:, :]
        elif data_class == "people":
            trainData = one_hot[0:self.number_data*210, :]
            testData = one_hot[self.number_data*210:, :]
        return trainData,testData

    def deal(self,dataOrigin):
        Row = dataOrigin.shape[0]
        Data = []
        for i in range(Row):
            RMS = self.sumRMS(dataOrigin[i,:])
            MAV = self.sumMAV(dataOrigin[i,:])
            WL = self.sumWL(dataOrigin[i,:])
            ZC = self.sumZC(dataOrigin[i,:])
            DASDV = self.sumDASDV(dataOrigin[i,:])
            LOG = self.sumLOG(dataOrigin[i,:])
            SII = self.sumSII(dataOrigin[i,:])
            TM3 = self.sumTM3(dataOrigin[i,:])
            TM4 = self.sumTM4(dataOrigin[i,:])
            TM5 = self.sumTM5(dataOrigin[i,:])
            frequencyRatio = self.frequencyRatio(dataOrigin[i,:])
            MFMN = self.sumMFMN(dataOrigin[i,:])
            MFMD = self.sumMFMD(dataOrigin[i,:])
            IEMG = self.sumIEMG(dataOrigin[i,:])
            Data_feature = [RMS,MAV,WL,ZC,DASDV,LOG,SII,TM3,TM4,TM5,frequencyRatio,MFMN,MFMD,IEMG]
            Data.append(Data_feature)
        Data = np.array(Data).reshape((-1,self.feature*self.channal))
        return Data

if __name__ == "__main__":
    data_import = DealSign()
    data = data_import.readFile("person")