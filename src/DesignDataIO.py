#!/usr/bin/env python

"""
License and copyright TBD soon

Author: Quentin Gautier
"""

from __future__ import division

import logging
import numpy as np



# Other imports defined in classes below:
#
# from scipy.io import loadmat
# import pandas as pd
# import matplotlib.pyplot as plt
# import warnings
# import matplotlib.cbook
#


logger = logging.getLogger(__name__)


class DesignDataBase(object):
    """
    Base class for loading design data.
    """
    def __init__(self):
        pass

    def getNumDesigns(self):
        return 0

    def getNumKnobs(self):
        return 0

    def getKnobSettings(self):
        return None

    def allIndexes(self):
        return np.arange(self.getNumDesigns())

    def hasGroundTruth(self):
        return False

    def getGroundTruth(self, indexes=None):
        """
        Return a MxN matrix with M the number of indexes and N the number of objectives
        """
        return None

    def hasHints(self):
        """
        Return a boolean array (True/False for each objective)
        """
        return [False]

    def hasHintsOnAllObjectives(self):
        return np.all(self.hasHints())

    def getHints(self, indexes=None):
        """
        Return a MxH matrix with M the number of indexes and H the number of available hint objectives
        (The array is NOT padded to match the number of objectives)
        """
        return None

    def __str__(self):
        return "| " + str(self.getNumDesigns()) + " designs with " + str(self.getNumKnobs()) + " knobs |"


class MatlabDesignData(DesignDataBase):
    """
    Load FPGA design data with ground truth from Matlab.
    Also loads GPU hints if it exists (can be replaced by estimated throughput).
    """
    
    def __init__(self, filename, hintType="gpu", normalizeHints=True):
        
        from scipy.io import loadmat

        super(MatlabDesignData, self).__init__()
        self.allDesigns = loadmat(filename)
        self.allDesigns['gt_Perf']    = 1 / self.allDesigns['run_results_timing']
        self.allDesigns['gt_Logic']   = 1 / self.allDesigns['logic_util']
        self.hasHints_ = np.zeros(2, dtype=bool)
        try:
            self.gpuHints = loadmat(filename[:-4] + "_wGPU_k20c.mat")
        except:
            self.gpuHints = self.allDesigns
            
        try:
            estThroughput = loadmat(filename[:-4] + "_est_tp.mat")
        except:
            estThroughput = self.allDesigns
        
        hintType = hintType.lower()
        
        if 'logic_util_synth_report' in self.gpuHints:
            self.hasHints_[1] = True
            self.gpuHints['hint_Logic'] = 1 / self.gpuHints['logic_util_synth_report']
        
        if 'run_results_timing_k20c' in self.gpuHints:
            self.gpuHints['hint_Perf_gpu'] = 1 / self.gpuHints['run_results_timing_k20c']
            if hintType == "gpu":
                self.hasHints_[0] = True
                self.gpuHints['hint_Perf'] = self.gpuHints['hint_Perf_gpu']
            
        if 'run_results_timing_esp_tp' in estThroughput:
            self.gpuHints['hint_Perf_esttp'] = 1 / estThroughput['run_results_timing_esp_tp']
            if hintType == "esttp":
                self.hasHints_[0] = True
                self.gpuHints['hint_Perf'] = self.gpuHints['hint_Perf_esttp']
            
        if 'time_cpu' in self.allDesigns:
            self.gpuHints['hint_Perf_cpu'] = 1 / self.allDesigns['time_cpu']
            if hintType == "cpu":
                self.hasHints_[0] = True
                self.gpuHints['hint_Perf'] = self.gpuHints['hint_Perf_cpu']
        
        if normalizeHints:
            for h in ['hint_Perf', 'hint_Logic', 'hint_Perf_gpu', 'hint_Perf_esttp', 'hint_Perf_cpu']:
                if h in self.gpuHints:
                    self.gpuHints[h] -= np.nanmin(self.gpuHints[h])
                    self.gpuHints[h] /= np.nanmax(self.gpuHints[h])


    def setIndexesToKeep(self, indexes):
        """
        Only keep data with the given indexes.
        Warning: Change the indexes.
        """
        self.allDesigns['gt_Perf']       = self.allDesigns['gt_Perf'][indexes]
        self.allDesigns['gt_Logic']      = self.allDesigns['gt_Logic'][indexes]
        self.allDesigns['knob_settings'] = self.allDesigns['knob_settings'][indexes,:]
        for h in ['hint_Perf', 'hint_Logic', 'hint_Perf_gpu', 'hint_Perf_esttp', 'hint_Perf_cpu']:
            if h in self.gpuHints:
                self.gpuHints[h] = self.gpuHints[h][indexes]
        

    def getNumDesigns(self):
        return self.allDesigns['knob_settings'].shape[0]

    def getNumKnobs(self):
        return self.allDesigns['knob_settings'].shape[1]

    def getKnobSettings(self):
        return self.allDesigns['knob_settings']

    def hasGroundTruth(self):
        return True

    def getGroundTruth(self, indexes=None):
        if indexes is None: indexes = self.allIndexes()
        return np.hstack((
            self.allDesigns['gt_Perf'][indexes],
            self.allDesigns['gt_Logic'][indexes]))

    def hasHints(self):
        return self.hasHints_

    def getHints(self, indexes=None, allPerfTypes=False):
        if indexes is None: indexes = self.allIndexes()
        hints = np.array([]).reshape(len(indexes),0)
        if self.hasHints_[0]:
            if allPerfTypes:
                for h in ['hint_Perf_gpu','hint_Perf_esttp','hint_Perf_cpu']:
                    if h in self.gpuHints: hints = np.c_[hints,self.gpuHints[h][indexes]]
            else:
                hints = self.gpuHints['hint_Perf'][indexes]
        if self.hasHints_[1]:
            hints = np.c_[hints, self.gpuHints['hint_Logic'][indexes]]
        return hints if len(hints) > 0 else None


class CsvDesignData(DesignDataBase):
    """
    Read objective and knob data from a CSV file.
    """

    def __init__(self, filename,
                 knob_names = None,
                 objectives = None,
                 obj_inv = True,
                 hint = None,
                 normalize_hint = True):
        
        import pandas as pd

        super(CsvDesignData, self).__init__()

        if objectives is None: objectives = ["time", "logic", "error"]
        if type(obj_inv) == bool: obj_inv = [obj_inv] * len(objectives)

        self.csv = pd.read_csv(filename)
        self.objectives = []
        self.hint = hint
        for o in objectives:
            if o not in list(self.csv):
                logger.warning("Warning: objective \"" + o + "\" not in CSV file")
                continue
            self.objectives.append(o)
        for i,o in enumerate(self.objectives):
                if obj_inv[i]: self.csv[o] = 1 / self.csv[o]
        if hint is not None:
            for h in hint:
                if h not in list(self.csv):
                    raise RuntimeError("Requested hint \"" + h + "\" not in file")
        if normalize_hint:
            yh = self.csv[self.hint]
            yh -= yh.min(axis=0)
            yh /= yh.max(axis=0)
            self.csv[self.hint] = yh
        if knob_names is not None:
            self.knob_names = knob_names
        else:
            self.knob_names = [name for name in self.csv if name.startswith("param") or name.startswith("knob")]
            

    def getNumDesigns(self):
        return self.csv[self.objectives[0]].shape[0]

    def getNumKnobs(self):
        return self.knob_settings.shape[1]

    def getKnobSettings(self):
        return np.array(self.csv[self.knob_names])

    def hasGroundTruth(self):
        return True

    def getGroundTruth(self, indexes=None):
        if indexes is None: indexes = self.allIndexes()
        return np.array(self.csv[self.objectives])[indexes]

    def hasHints(self):
        if self.hint is not None:
            return [True] * len(self.objectives)

    def getHints(self, indexes=None):
        if indexes is None: indexes = self.allIndexes()
        return np.array(self.csv[self.hint])[indexes]


class InputOutputData(DesignDataBase):
    """
    Wrapper around input matrix X, output matrix y, and optionally hint output yh.
    """
    def __init__(self, X, y, yh=None):
        super(InputOutputData, self).__init__()
        self.X = X
        self.y = y
        self.yh = yh

    def getNumDesigns(self):
        return self.X.shape[0]

    def getNumKnobs(self):
        return self.X.shape[1]

    def getKnobSettings(self):
        return self.X

    def hasGroundTruth(self):
        return True

    def getGroundTruth(self, indexes=None):
        """
        Return a MxN matrix with M the number of indexes and N the number of objectives
        """
        if indexes is None: indexes = self.allIndexes()
        return self.y[indexes,:]

    def hasHints(self):
        n_obj = 1
        if self.y.ndim > 1: n_obj = self.y.shape[1]
        return [self.yh is not None]*n_obj

    def getHints(self, indexes=None):
        return self.yh





class DataSaver(object):
    """
    Base class for saving or processing the data from the algorithm
    """
    def __init__(self):
        pass

    def setDesigns(self, designs):
        pass

    def setKnobs(self, knobs):
        pass

    def setSampledIndexes(self, indexes):
        pass

    def setRelaxedIndexes(self, indexes):
        pass

    def setPredictionsIndexes(self, indexes):
        pass

    def setPredictions(self, predictions, estimators = None):
        pass

    def setPredictedDistancesInGroundTruthSet(self, pred_distances, gt_distances):
        pass

    def addPredictedDistanceVector(self, distance):
        pass

    def setHintEliminatedIndexes(self, indexes):
        pass

    def setEliminationWeight(self, weigths):
        pass

    def setAlgorithmStart(self):
        pass

    def setIterationDone(self):
        pass

    def setAlgorithmDone(self):
        pass


class DataPlotter2D(DataSaver):
    """
    Plot the results of the algorithm along two axes
    """    

    def __init__(self,
            blocking        = False,
            plotPredictions = False,
            plotDistances   = False,
            plotHintSpace   = False):
        super(DataPlotter2D, self).__init__()
 
        import matplotlib.pyplot as plt
        # Suppress matplotlib deprecation warning
        import warnings
        import matplotlib.cbook
        warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

        self.plt = plt
       
        self.reset()
        self.doPlotPredictions = plotPredictions
        self.doPlotDistances   = plotDistances
        self.doPlotHintSpace   = plotHintSpace
        self.blocking = blocking
        self.numPlots = (1 + int(self.doPlotPredictions)
                + 2*int(self.doPlotDistances)
                + int(self.doPlotHintSpace))
        self.fig = self.plt.figure()
        self.ax  = []
        for i in range(self.numPlots):
            self.ax.append(self.fig.add_subplot(
                np.floor(np.sqrt(self.numPlots)),
                np.ceil(self.numPlots / np.floor(np.sqrt(self.numPlots))),
                i+1))
        self.plt.ion()

    def reset(self):
        self.designs     = None
        self.groundTruth = None
        self.hintSpace   = None
        self.predictions = None
        self.estimators  = None
        self.selectedDistances  = []
        self.gtDistances        = []
        self.predictedDistances = []
        self.sampledIndexes     = []
        self.relaxedIndexes     = []
        self.predictionsIndexes = []
        self.hintEliminatedIndexes = np.array([],dtype=np.int)
        self.elimWeights = None

    def setDesigns(self, designs):
        self.designs  = designs
        self.allKnobs = self.designs.getKnobSettings()
        if self.designs.hasGroundTruth():
            groundTruth = self.designs.getGroundTruth(range(self.designs.getNumDesigns()))
            groundTruthMax = np.amax(groundTruth, axis=0)
            self.groundTruth = groundTruth / groundTruthMax
        if self.designs.hasHintsOnAllObjectives():
            self.hintSpace = self.designs.getHints()[:,:2]
            self.hintSpace = self.hintSpace / np.nanmax(self.hintSpace,0)

    def setKnobs(self, knobs):
        self.allKnobs = knobs

    def setSampledIndexes(self, indexes):
        self.sampledIndexes = indexes

    def setRelaxedIndexes(self, indexes):
        self.relaxedIndexes = indexes

    def setPredictionsIndexes(self, indexes):
        if self.doPlotPredictions:
            self.predictionsIndexes = indexes

    def setPredictions(self, predictions, estimators = None):
        if self.doPlotPredictions:
            self.predictions = predictions
            self.estimators = estimators

    def setPredictedDistancesInGroundTruthSet(self, pred_distances, gt_distances):
        if self.doPlotDistances:
            self.selectedDistances.append(pred_distances)
            self.gtDistances.append(gt_distances)

    def addPredictedDistanceVector(self, distance):
        if self.doPlotDistances:
            self.predictedDistances.append(distance)

    def setHintEliminatedIndexes(self, indexes):
        self.hintEliminatedIndexes = np.append(self.hintEliminatedIndexes, indexes)

    def setEliminationWeight(self, weigths):
        self.elimWeights = weigths
    
    def setAlgorithmStart(self):
        self.__updatePlot()

    def setIterationDone(self):
        self.__updatePlot()

    def setAlgorithmDone(self):
        self.__updatePlot()

    def __updatePlot(self):
        plotIdx = 0

        # Plot sampling over ground truth
        if self.groundTruth is not None: 
            self.ax[plotIdx].clear()
            self.ax[plotIdx].set_xlim([0,1.05])
            self.ax[plotIdx].set_ylim([0,1.05])
            self.ax[plotIdx].set_title("ATNE sampling")
            self.ax[plotIdx].plot(self.groundTruth[:,0], self.groundTruth[:,1], 'x', color="0.7", markeredgewidth=1.8, markersize=5)
            if len(self.hintEliminatedIndexes) > 0 and self.elimWeights is None:
                self.ax[plotIdx].plot(self.groundTruth[self.hintEliminatedIndexes,0], self.groundTruth[self.hintEliminatedIndexes,1], 'x', color="indianred", markeredgewidth=1.8, markersize=5)
            if self.elimWeights is None:
                self.ax[plotIdx].plot(self.groundTruth[self.relaxedIndexes,0], self.groundTruth[self.relaxedIndexes,1], 'x', color="g", markeredgewidth=1.8, markersize=5)
            else:
                #self.ax[plotIdx].plot(self.groundTruth[self.sampledIndexes,0], self.groundTruth[self.sampledIndexes,1], 'o', color="b", markeredgewidth=1.8, markersize=5, alpha=0.6)
                weights = np.sum(np.mean(self.elimWeights,axis=1) / np.max(np.mean(self.elimWeights,axis=1),axis=0),axis=1)
#                 weights = np.mean(self.elimWeights[:,:,0], axis=1)
                alpha = 1 - (weights / np.max(weights)) / 2
                red = weights / np.max(weights)
                for i in range(self.groundTruth.shape[0]):
                    if i not in self.relaxedIndexes: continue 
                    if np.isnan(red[i]): red[i] = 0
                    self.ax[plotIdx].plot(self.groundTruth[i,0], self.groundTruth[i,1], 'x', color=[red[i],1-red[i],0], markeredgewidth=1.8, markersize=5)
            self.ax[plotIdx].plot(self.groundTruth[self.sampledIndexes,0], self.groundTruth[self.sampledIndexes,1], 'x', color="b", markeredgewidth=1.8, markersize=5)
            plotIdx += 1

        # Plot predicted design space
        if self.predictions is not None and self.doPlotPredictions:
            self.ax[plotIdx].clear()
            

            labeledMask = np.in1d(self.predictionsIndexes, self.sampledIndexes)
            labeledMaskIdx = np.where(labeledMask)[0]
            cmap = self.plt.cm.get_cmap('hsv')
            shapes = ['x', '.', '+']

            # Plot type 1
            #self.ax[plotIdx].set_title("Estimated design spaces by each forest")
            #for f in range(self.predictions.shape[0]):
            #    self.ax[plotIdx].plot(self.predictions[f,:,0], self.predictions[f,:,1], 'x', markeredgewidth=1.8, markersize=5)
            #    #self.ax[plotIdx].plot(self.predictions[f,labeledMask,0], self.predictions[f,labeledMask,1], 'x', markeredgewidth=1.8, markersize=5)

            # Plot type 2
            #import matplotlib
            #for i,p in enumerate(labeledMaskIdx):
            #    color = cmap(i/len(labeledMaskIdx))
            #    predmean = self.predictions[:,p,:].mean(0)
            #    predmed  = np.median(self.predictions[:,p,:], 0)
            #    predstd  = self.predictions[:,p,:].std(0)

            #    # Plot type 2.1
            #    #self.ax[plotIdx].plot(self.predictions[:,p,0], self.predictions[:,p,1], shapes[i%len(shapes)], markeredgewidth=1.8, markersize=5, color=color)
            #    #self.ax[plotIdx].plot(predmean[0], predmean[1], shapes[0], markeredgewidth=1.8, markersize=5, color=color)
            #    #self.ax[plotIdx].plot(predmed[0], predmed[1], shapes[1], markeredgewidth=1.8, markersize=5, color=color)

            #    # Plot type 2.2
            #    circle = matplotlib.patches.Ellipse(predmean[[0,1]], predstd[0], predstd[1])
            #    self.ax[plotIdx].add_artist(circle)


            # Plot type 3 (Mean predictions)
#             self.ax[plotIdx].set_title("Average estimated P_relaxed")
#             pred_mean = self.predictions.mean(0)
#             self.ax[plotIdx].plot(pred_mean[:,0], pred_mean[:,1], 'x', markeredgewidth=1.8, markersize=5)


            # Plot type 4 (Mean predictions of the entire space)
            self.ax[plotIdx].set_title("Average estimated design space")
            if self.estimators is not None:
                predictions = np.empty([self.predictions.shape[0], self.designs.getNumDesigns(), self.predictions.shape[2]])
                for f in range(self.predictions.shape[0]):
                    for o in range(self.predictions.shape[2]):
                        predictions[f,:,o] = self.estimators[f][o].predict(self.allKnobs)
                pred_mean = predictions.mean(0)
                self.ax[plotIdx].scatter(pred_mean[:,0], pred_mean[:,1], marker='x', c=np.arange(pred_mean.shape[0])/pred_mean.shape[0])

                # Some tests here, although I can't remember what I was testing exactly...
                if False:
                    from nonconformist.cp import IcpRegressor
                    from nonconformist.nc import NcFactory
                    from sklearn.ensemble import RandomForestRegressor
                    
                    model1 = RandomForestRegressor()
                    nc1    = NcFactory.create_nc(model1)
                    icp1   = IcpRegressor(nc1)
 
                    model2 = RandomForestRegressor()
                    nc2    = NcFactory.create_nc(model2)
                    icp2   = IcpRegressor(nc2)
 
                    n = self.sampledIndexes.size
 
                    idx = np.random.permutation(n)
                    idx_train, idx_cal = idx[:int(0.8*n)], idx[int(0.8*n):]
 
                    icp1.fit(self.allKnobs[self.sampledIndexes][idx_train,:], self.groundTruth[self.sampledIndexes,0][idx_train])
                    icp2.fit(self.allKnobs[self.sampledIndexes][idx_train,:], self.groundTruth[self.sampledIndexes,1][idx_train])
                     
                    icp1.calibrate(self.allKnobs[self.sampledIndexes][idx_cal,:], self.groundTruth[self.sampledIndexes,0][idx_cal])
                    icp2.calibrate(self.allKnobs[self.sampledIndexes][idx_cal,:], self.groundTruth[self.sampledIndexes,1][idx_cal])
 
                    prediction1 = icp1.predict(self.allKnobs, significance=0.05)
                    prediction2 = icp2.predict(self.allKnobs, significance=0.05)
                    
                    print(prediction1)
                    
                    self.ax[plotIdx].errorbar(pred_mean[:,0], pred_mean[:,1], xerr=prediction1 , yerr=prediction2, linestyle="None")





            # Keep this
            #self.ax[plotIdx].set_xlim(left=0, right=2)
            #self.ax[plotIdx].set_ylim(bottom=0, top=2)
            plotIdx += 1

        # Plot hint space if available
        if self.doPlotHintSpace and self.hintSpace is not None:
            self.ax[plotIdx].clear()
            self.ax[plotIdx].set_xlim([0,1.05])
            self.ax[plotIdx].set_ylim([0,1.05])
            self.ax[plotIdx].set_title("Hint space")
            self.ax[plotIdx].plot(self.hintSpace[:,0], self.hintSpace[:,1], 'x', markeredgewidth=1.8, markersize=5)
            plotIdx += 1


        # Plot distances for labeled samples
        if self.selectedDistances and self.doPlotDistances:
            self.ax[plotIdx].clear()
            self.ax[plotIdx].set_title("Estimated distances for labeled samples")
            for d in self.selectedDistances:
                self.ax[plotIdx].hist(d.flatten(), 50, alpha=0.65)
                #self.ax[plotIdx].hist(d.mean(0), 50, alpha=0.65)
            #for d in self.gtDistances:
            #    self.ax[plotIdx].hist(d.flatten(), 50, alpha=0.65)
            plotIdx += 1

        # Plot distances for unlabeled samples
        if self.predictedDistances and self.doPlotDistances:
            self.ax[plotIdx].clear()
            self.ax[plotIdx].set_title("Estimated distances for unlabeled samples (within P_relaxed)")
            predictedDistances = np.array(self.predictedDistances)
            for d in range(predictedDistances.shape[2]):
                self.ax[plotIdx].hist(predictedDistances[:,:,d].flatten(), 50, alpha=0.65)
            plotIdx += 1

        self.plt.show()
        try: self.plt.pause(0.00001)
        except: pass
        self.fig.canvas.draw()
        if self.blocking:
            self.fig.waitforbuttonpress()

        self.selectedDistances  = []
        self.gtDistances        = []
        self.predictedDistances = []


    def block(self):
        self.plt.ioff()
        self.plt.show()





