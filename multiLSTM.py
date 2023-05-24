import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from sklearn.metrics import r2_score
import keras as keras
import tensorflow as tf
np.random.seed(1234)
from keras import backend as K

class multiLSTM:
    def __init__(self):
        self.inputHorizon = 12  # number of time steps as input
        self.inOutVecDim = 15   # number of stations
        self.lstmModels = [ None for _ in range(6)]
        self.xTest, self.yTest = None, None
        file_dataset = r'C:\Users\USER\DATA_SET2_long.dat'
        with open(file_dataset) as f:
            data = csv.reader(f, delimiter=",")
            winds = []
            for line in data:
                winds.append((line))
        self.winds = (np.array(winds)).astype(float)    # all data
        self.winds = self.winds[:,:self.inOutVecDim]
        self.means_stds = [0,0]
        self.winds, self.means_stds = self.normalize_winds_0_1(self.winds)
        self.validation_split = 0.05    #float b/w 0 and 1: stands for the fraction of data to be used for validation
        self.batchSize = 3              # number of samples that will be propagated through the network
        activation = ['sigmoid',   "tanh",   "relu", 'linear']
        self.activation = activation[2]
        realRun = 1
        #          model number :           1   2   3   4   5   6
        self.epochs, self.trainDataRate = [[15, 17, 15, 17, 15, 15], 1] if realRun else [[ 1, 1, 1, 1, 1, 1] , 0.005]   # percentage of data used for training(saving time for debugging)

    def normalize_winds_0_1(self, winds):
        '''normalize based on each station data'''
        stations = winds.shape[1] # shape returns (rows, columsn) so shape[1] is column numbers
        normal_winds = []
        mins_maxs = []
        windMax = winds.max()
        windMin = winds.min()
        normal_winds = (winds - windMin) / windMax
        mins_maxs = [windMin, windMax]
        return np.array(normal_winds), mins_maxs

    def denormalize(self, vec): #used to denormalise Ytest
        res = vec * self.means_stds[1] + self.means_stds[0]        #  from 0 to 1
        return res

    def loadData_1(self):
        # for lstm1 output xtrain ytrain
        result = []
        for index in range(len(self.winds) - self.inputHorizon):
            result.append(self.winds[index:index + self.inputHorizon])
        result = np.array(result)  

        trainRow = int(4500 * self.trainDataRate)
        X_train = result[:trainRow, :]
        y_train = self.winds[self.inputHorizon:trainRow + self.inputHorizon]
        self.xTest = result[4500:4861, :]
        self.yTest = self.winds[4500 + self.inputHorizon:4861 + self.inputHorizon]
        self.predicted = np.zeros_like(self.yTest)
        return [X_train, y_train]

    def loadData(self, preXTrain, preYTrain, model): # xtrain and ytrain from loadData_1
        # for lstm2 output: xtrain ytrain
        xTrain, yTrain = np.ones_like(preXTrain), np.zeros_like(preYTrain)
  
        for ind in range(len(preXTrain) - self.inputHorizon -1):
            tempInput = preXTrain[ind]
            temp_shape = tempInput.shape
            tempInput = np.reshape(tempInput, (1,temp_shape[0],temp_shape[1]))
            output = model.predict(tempInput)
            tInput = np.reshape(tempInput,temp_shape)
            tempInput = np.vstack((tInput, output))
            tempInput = np.delete(tempInput, 0, axis=0)
            xTrain[ind] = tempInput
            yTrain[ind] = preYTrain[ind+1]
        return [xTrain, yTrain]

  
    def buildModelLSTM_1(self):
        model = Sequential()
        in_nodes = out_nodes = self.inOutVecDim
        layers = [in_nodes, 15*2, 15, 32, out_nodes]
        model.add(LSTM(input_dim=layers[0],output_dim=layers[1],
            return_sequences=False))
    
        model.add(Dense(
            output_dim=layers[4]))
        model.add(Activation(self.activation))
    
        optimizer = keras.optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM_2(self):
        model = Sequential()
        layers = [self.inOutVecDim, 10 , 15 * 2, 32, self.inOutVecDim]
        model.add(LSTM(input_dim=layers[0],output_dim=layers[1],
            return_sequences=False))

        model.add(Dense(
            output_dim=layers[4]))

        model.add(Activation(self.activation))

        optimizer = keras.optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM_3(self):
        model = Sequential()

        layers = [self.inOutVecDim, 15, 15 * 2, 32, self.inOutVecDim]
        model.add(LSTM(input_dim=layers[0], output_dim=layers[1],
            return_sequences=False))

        model.add(Dense(
            output_dim=layers[4]))

        model.add(Activation(self.activation))

        optimizer = keras.optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM_4(self):
        model = Sequential()

        layers = [self.inOutVecDim, 15, 15 * 2, 15, self.inOutVecDim]
        model.add(LSTM(input_dim=layers[0], output_dim=layers[1],
            return_sequences=True))

        model.add(LSTM(layers[2],
            return_sequences=False))

        model.add(Dense(output_dim=layers[4]))

        model.add(Activation(self.activation))

        optimizer = keras.optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM_5(self):
        model = Sequential()

        layers = [self.inOutVecDim, 30, 15 * 2, 15, self.inOutVecDim]
        model.add(LSTM(input_dim=layers[0], output_dim=layers[1],
            return_sequences=False))

        model.add(Dense(output_dim=layers[4]))

        model.add(Activation(self.activation))

        optimizer = keras.optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM_6(self):
        model = Sequential()
        layers = [self.inOutVecDim, 15*2, 15 * 2, 15, self.inOutVecDim]
        model.add(LSTM(input_dim=layers[0], output_dim=layers[1],
        return_sequences=True))


        model.add(LSTM(layers[2],
            return_sequences=False))

        model.add(Dense(output_dim=layers[4]))

        model.add(Activation(self.activation))

        optimizer = keras.optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM(self, lstmModelNum):
        if   lstmModelNum == 1:
            return self.buildModelLSTM_1()
        elif lstmModelNum == 2:
            return self.buildModelLSTM_2()
        elif lstmModelNum == 3:
            return self.buildModelLSTM_3()
        elif lstmModelNum == 4:
            return self.buildModelLSTM_4()
        elif lstmModelNum == 5:
            return self.buildModelLSTM_5()
        elif lstmModelNum == 6:
            return self.buildModelLSTM_6()

    def trainLSTM(self, xTrain, yTrain, lstmModelNum):
        # train first LSTM with inputHorizon number of real input values

        lstmModel = self.buildModelLSTM(lstmModelNum)
        lstmModel.fit(xTrain, yTrain, batch_size=self.batchSize, nb_epoch=self.epochs[lstmModelNum-1], validation_split=self.validation_split)
        return lstmModel

    def test(self):
        ''' calculate the predicted values(self.predicted) '''
        for ind in range(len(self.xTest)):
            modelInd = ind % 6
            if modelInd == 0:
                testInputRaw = self.xTest[ind]
                testInputShape = testInputRaw.shape
                testInput = np.reshape(testInputRaw, [1, testInputShape[0], testInputShape[1]])
            else :
                testInputRaw = np.vstack((testInputRaw, self.predicted[ind-1]))
                testInput = np.delete(testInputRaw, 0, axis=0)
                testInputShape = testInput.shape
                testInput = np.reshape(testInput, [1, testInputShape[0], testInputShape[1]])

            self.predicted[ind] = self.lstmModels[modelInd].predict(testInput)

        return

    def errorMeasures(self, denormalYTest, denormalYPredicted):

        mae = np.mean(np.absolute(denormalYTest - denormalYPredicted))
        rmse = np.sqrt((np.mean((np.absolute(denormalYTest - denormalYPredicted)) ** 2)))
        nrsme_maxMin = 100*rmse / (denormalYTest.max() - denormalYTest.min())
        nrsme_mean = 100 * rmse / (denormalYTest.mean())
        corr_2 = r2_score(denormalYTest, denormalYPredicted)
        
        return mae, rmse,nrsme_maxMin, nrsme_mean, corr_2
    
    def drawGraphStation(self, station, visualise = 1, ax = None ):
        '''draw graph of predicted vs real values'''

        yTest = self.yTest[:, station]
        denormalYTest = self.denormalize(yTest)

        denormalPredicted = self.denormalize(self.predicted[:, station])
        
        mae, rmse,nrsme_maxMin ,nrmse_mean, corr_2  = self.errorMeasures(denormalYTest, denormalPredicted)
        print ('station %s : MAE = %7.7s   RMSE = %7.7s  nrsme_maxMin = %7.7s    nrmse_mean = %7.7s R^2 score = %7.7s'%(station+1, mae, rmse, nrsme_maxMin,nrmse_mean, corr_2 ))

        if visualise:
            if ax is None :
                fig = plt.figure()
                ax = fig.add_subplot(111)

            ax.plot(denormalYTest, label='Real')
            ax.plot(denormalPredicted, label='Predicted', color='red')

        return mae, rmse, nrsme_maxMin ,nrmse_mean, corr_2

    def drawGraphAllStations(self, staInd):
        rows, cols = 1,1 
        fig, ax_array = plt.subplots(rows, cols, sharex=True, sharey=True )
        maeRmse = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for ax in np.ravel(ax_array):
            maeRmse[staInd] = self.drawGraphStation(staInd, visualise=1, ax=ax)
            plt.xticks([0, 100, 200, 300])
           
        filename = 'KSAlongEpochSt '+ str(staInd)
        plt.savefig('{}.png'.format(filename))
        plt.savefig('{}.png'.format(filename))
        plt.show()

        return

    def run(self):
        #  training
        xTrain, yTrain = self.loadData_1()
        print (' Training LSTM 1 ...')
        self.lstmModels[0] = self.trainLSTM(xTrain, yTrain, 1)

        for modelInd in range(1,6):
            xTrain, yTrain = self.loadData(xTrain, yTrain, self.lstmModels[modelInd-1])
            print (' Training LSTM %s ...' % (modelInd+1))
            self.lstmModels[modelInd] = self.trainLSTM(xTrain, yTrain, modelInd+1)

        # testing
        print ('...... TESTING  ...')
        self.test()
        for staInd in range(15):
            self.drawGraphAllStations(staInd)

DeepForecaste = multiLSTM()
DeepForecaste.run()
