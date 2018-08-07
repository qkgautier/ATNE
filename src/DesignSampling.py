#!/usr/bin/env python

"""
License and copyright TBD soon

Authors:
    - Original ATNE Matlab code: Pingfan Meng
    - Python port, modifications and additions: Quentin Gautier
"""

from __future__ import division

import numpy as np

import scipy
from scipy.spatial.distance import pdist, squareform
from scipy.special import comb, erfinv

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import Imputer
from sklearn.neighbors.kde import KernelDensity

import logging
import signal
from timeit import default_timer as timer
import multiprocessing as mp

import DesignDataIO
import Distances


logger = logging.getLogger(__name__)


class DesignSampler(object):
    """
    Base class for sampling algorithms
    """
    def __init__(self, designs = DesignDataIO.DesignDataBase()):
        self.setDesignData(designs)
        self.reset()

    def setDesignData(self, designs):
        self.designs = designs
    
    def reset(self):
        self.sampledIndexes = np.empty(0, dtype=np.intc)

    def run(self):
        pass

    def getSampledIndexes(self):
        return self.sampledIndexes



class TED(DesignSampler):
    """
    Implement TED sampling algorithm
    """
    
    TED_ONLY            = 0
    TED_AND_HINT_PARETO = 1
    TED_HINT_OUTPUT     = 2
    TED_ALL_HINT_OUTPUT = 3
    
    def __init__(self,
            designs = DesignDataIO.DesignDataBase(),
            numSamples = 20,
            sigma = 0,
            lamb = 10e-4,
            method=TED_ONLY,
            minTEDRatio = 0.5,
            enableStats = False):
        super(TED, self).__init__(designs)
        self.setNumSamples(numSamples)
        self.setSigma(sigma)
        self.setLambda(lamb)
        self.setMethod(method)
        self.setMinTEDRatio(minTEDRatio)
        self.setEnableStats(enableStats)

    def setNumSamples(self, numSamples):
        self.numSamples = numSamples

    def setSigma(self, sigma):
        self.sigma = sigma

    def setLambda(self, lamb):
        self.lamb = lamb
    
    def setMinTEDRatio(self, ratio):
        self.minTEDRatio = ratio
        
    def setMethod(self, method):
        self.method = method
       
    def setEnableStats(self, enable):
        self.enableStats = enable

    def run(self):
        
        logger.info("Running TED method #" + str(self.method))
        
        numDesigns    = self.designs.getNumDesigns()
        numTEDSamples = self.numSamples
        
        self.sampledIndexes = np.empty(0, dtype=np.uint64)
        
        # Operate on the whole design space
        remainingIndexes = np.arange(numDesigns, dtype=np.uint64)
        
        # Use knob settings as the input X matrix
        tedInput = self.designs.getKnobSettings()
        
        
        if self.method == TED.TED_AND_HINT_PARETO:
            hints         = self.designs.getHints()
            _, paretoIndexes, paretoScores = Distances.prpt(hints)
            paretoIndexes = paretoIndexes.astype(np.uint64)
            numPareto     = paretoIndexes.size
            
            numPareto = int(min(numPareto, self.numSamples * (1-self.minTEDRatio)))
            numTEDSamples -= numPareto
            
            # Choose some Pareto designs from the hint space
            self.sampledIndexes = paretoIndexes[np.argsort(paretoScores)[::-1]][:numPareto]
            
            # Remove these designs from the TED input matrix 
            remainingIndexes = np.setdiff1d(remainingIndexes, self.sampledIndexes)
            tedInput = self.designs.getKnobSettings()[remainingIndexes]
            
        elif self.method == TED.TED_HINT_OUTPUT:
            # Set the hint output (objective values) as the TED input matrix
            hints = self.designs.getHints()
            tedInput = hints
            
        elif self.method == TED.TED_ALL_HINT_OUTPUT:
            tedInput = self.designs.getHints(allPerfTypes=True)
            
        
        if self.sigma == 0:
            # A reasonable choice of sigma is on the order of the total range.
            self.sigma = np.sqrt(np.max(scipy.ptp(self.designs.getKnobSettings(),0)))
 
        
        # Run TED
        tedIndexes = self.transdesign_sq(tedInput, numTEDSamples, self.sigma, self.lamb)

        self.sampledIndexes = np.r_[self.sampledIndexes, remainingIndexes[tedIndexes]]

        if self.enableStats:
            stats = []
            for i in range(1, self.sampledIndexes.size+1):
                if self.method == TED.TED_AND_HINT_PARETO:
                    numPareto_i = int(min(paretoIndexes.size, i * (1-self.minTEDRatio)))
                    numTEDSamples_i = i - numPareto_i
                    sampledIdx = np.r_[self.sampledIndexes[:numPareto_i], self.sampledIndexes[numPareto:numPareto+numTEDSamples_i]]
                else:
                    sampledIdx = self.sampledIndexes[:i]
                adrs = Distances.adrs(self.designs.getGroundTruth(), sampledIdx)
                stats.append([i, i / self.designs.getNumDesigns(), adrs])
            self.stats = np.array(stats)

    '''
    TED
    inputs:
        X - Matrix of all data samples
            (NxD matrix, where N is the number of items and D is their dimensionality)
        m - num of data samples to select
        sigma - variance?
        lambda - regularization parameter
    output:
        index - indices of selected data samples
    '''
    def transdesign_sq(self, X, m, sigma, lamb=10e-4):
        pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
        K = np.matrix(scipy.exp(-pairwise_sq_dists / (2*sigma**2)))

        candidate_index = list(range(K.shape[0]))

        n = K.shape[0]
        m = min(n,m)

        index = np.zeros(m, dtype=np.intc)
        for i in range(m):
            score = [0]*len(candidate_index)
            for j in range(len(candidate_index)):
                k = candidate_index[j]
                score[j] = K[k,:]*K[:,k] / (K[k,k] + lamb)
            I = np.argmax(score)
            index[i] = candidate_index[I]
            del candidate_index[I]
            K -= K[:,index[i]]*K[index[i],:] / (K[index[i],index[i]] + lamb)
        return index


class HintParetoSampler(DesignSampler):
    """
    Sample designs based on hint data.
    """
    def __init__(self,
            designs = DesignDataIO.DesignDataBase(),
            margin=0):
        super(HintParetoSampler, self).__init__(designs)
        self.setMargin(margin)

    def setMargin(self, margin):
        self.margin = margin
        
    def run(self):
        hints         = np.nan_to_num(self.designs.getHints())
        paretoIndexes = Distances.approximate_pareto(hints, margin=self.margin)[1].astype(np.uint64)
        
        self.sampledIndexes = paretoIndexes


class GroundTruthParetoSampler(DesignSampler):
    """
    Sample Pareto designs from the ground truth.
    """
    def __init__(self,
            designs = DesignDataIO.DesignDataBase(),
            inverse=False):
        super(GroundTruthParetoSampler, self).__init__(designs)
        self.setInversePareto(inverse)
        
    def setInversePareto(self, inverse):
        self.inverse = inverse
        
    def run(self):
        if self.inverse:
            designSpace   = 1/self.designs.getGroundTruth()
        else:
            designSpace   = self.designs.getGroundTruth()
        paretoMask    = Distances.getParetoOptimalDesigns(designSpace)
        paretoIndexes = np.where(paretoMask)[0].astype(np.uint64)
        
        self.sampledIndexes = paretoIndexes




###########################################################################
###########################################################################
###########################################################################



def atne_createAndFitEstimators(params):
    """
    ATNE code to generate and train one estimator per objective.
    """
    
    (numObjectives, numBootstrapSamples, numKnobsPerSplit, numTrees, ratioVarBootstrap, labeledKnobs, labels) = params
    
    estimators = np.empty([numObjectives], dtype=np.object_)
    
    #bootstrap_idx = np.random.randint(0, numLabeled, numBootstrapSamples)
    for o in range(numObjectives):
        
        ### Choice 1: Using bootstrap for each forest
        #estimator = RandomForestRegressor(n_estimators=self.numTrees, max_features=numKnobsPerSplit)
        #estimator.fit(labeledKnobs[bootstrap_idx], labels[bootstrap_idx, o])
        
        ### Choice 2: Using bootstrap for trees inside the forests
        # BaggingRegressor with DecisionTreeRegressor is the same as RandomForestRegressor,
        # but we can provide the max_samples parameter, which is the boostrap ratio for trees.
        estimator = BaggingRegressor(
            DecisionTreeRegressor(max_features=numKnobsPerSplit),
            n_estimators = numTrees,
            max_samples  = ratioVarBootstrap)
        
        estimator.fit(labeledKnobs, labels[:,o])
        
        estimators[o] = estimator
        
    return estimators


def atne_relaxedCheckEliminate(params):
    """
    ATNE elimination code for design xIdx. Check if relaxedIndexes[xIdx] should be eliminated.
    """

    (xIdx, numThresholds, numRelaxed, numObjectives, numForests, predictions, delta, relaxedIndexes, indexList, elimAgreeThreshold, dataSaver) = params

    isGreaterThanDelta = np.zeros([numThresholds, numRelaxed, numObjectives])

    # Calculate the difference (xp - x) for each estimator
    for f in range(numForests):
        
        distances = predictions[f,:,:] - predictions[f,xIdx,:]

        if dataSaver is not None:
            dataSaver.addPredictedDistanceVector(distances)

        # Note: Greater than delta: (xp - x > delta <=> x + delta < xp)
        # Note2: (distance > delta) is a boolean, converted to an int
        for i in range(numThresholds):
            isGreaterThanDelta[i] += (distances > delta[i]).astype(np.uint)

    # Percentage of estimators that think that (xp - x) > delta
    isGreaterThanDelta /= numForests

    # Make sure that there exists at least one 'xp' s.t. a percentage of the estimators agree that '(xp - x) > delta'
    # For smaller deltas, you need more number of 'xp' s.t. estimators agree that '(xp - x) > delta'
    if np.any(np.sum(np.all(isGreaterThanDelta[:,indexList,:] >= elimAgreeThreshold, 2), 1) > (range(1,numThresholds+1))):
        designToEliminateIdx = relaxedIndexes[xIdx]
        return designToEliminateIdx
    else:
        return -1


def atne_getSampleSelectionScore(params):
    """
    ATNE Sample selection code. Calculate the sample selection score for one design
    """

    (xIdx, predictions, unknownMaskP, delta) = params

    # Calculate difference between current point and all the other points in the set
    # (x - xp)
    distances = np.swapaxes(predictions[:,xIdx,:] - np.swapaxes(predictions[:,unknownMaskP,:],0,1), 0, 1)

    # Multiply the distances of each objective if the distance is greater than -delta on all objectives
    # Take the norm and sum all these values for all points across all forests
    return np.sum(np.abs( np.multiply(np.multiply.reduce(distances, 2), np.all(distances >= -delta[0], 2)) ))


def atne_hintEstimatedDistanceInCluster(params):
    """
    ATNE algorithm to estimate a distance sign within a cluster
    """
    
    (distIdx, hintDistances, hintClusters, hintClustersRatios, numObjectives, clusterRatioThresh) = params
    
    distCluster = hintClusters[:,distIdx]

    distClusterRatio = hintClustersRatios[np.arange(numObjectives), distCluster]
    
    # If this distance belongs to a cluster that is "pure" enough
    if np.all(np.max(distClusterRatio,1) >= clusterRatioThresh):

        # Get the correlation "direction" (i.e. the cluster is mostly composed of (1) or (-1))
        correlationDirection = np.ones(numObjectives)
        correlationDirection[distClusterRatio[:,0] > distClusterRatio[:,1]] = -1

        estimatedDistance = correlationDirection * hintDistances[:,distIdx]
        
        if np.all(estimatedDistance < 0):
            return np.int8(-1)
        elif np.all(estimatedDistance > 0):
            return np.int8(1)
    return np.int8(0)



class ATNE(DesignSampler):
    """
    Implements the ATNE algorithm.
    """
    def __init__(self,
            designs            = DesignDataIO.DesignDataBase(),
            initSampleAlgo     = DesignSampler(),
            sampleBudget       = 100,
            numForests         = 7,
            numTrees           = 10,
            ratioVarBootstrap  = 0.85,
            pdfThreshold       = 0.25,
            numDeltaThresholds = 4,
            elimAgreeThreshold = 0.4,
            minDeltaCandidates = 6,
            enableElimination  = True,
            randomSeed         = None,
            dataSaver          = DesignDataIO.DataSaver(),
            useHintIfAvailable = True,
            clusterRatioThresh = 1.0,
            clusterBeta        = 0.1,
            clusterEpsilon     = 0.25,
            hintEstimationThreshold = 2,
            hintValidationThreshold = 5,
            numWorkers         = 1,
            enableStats        = False
            ):
        super(ATNE, self).__init__(designs)
        self.pool = None
        self.setInitSampleAlgo(initSampleAlgo)
        self.setSampleBudget(sampleBudget)
        self.setNumForests(numForests)
        self.setNumTrees(numTrees)
        self.setRatioVarBootstrap(ratioVarBootstrap)
        self.setRandomSeed(randomSeed)
        self.setPdfThreshold(pdfThreshold)
        self.setNumDeltaThresholds(numDeltaThresholds)
        self.setElimAgreeThreshold(elimAgreeThreshold)
        self.setMinDeltaCandidates(minDeltaCandidates)
        self.setEliminationEnabled(enableElimination)
        self.setDataSaver(dataSaver)
        self.setNumWorkers(numWorkers)
        self.setUseHintIfAvailable(useHintIfAvailable)
        self.setClusterRatioThreshold(clusterRatioThresh)
        self.setClusterBeta(clusterBeta)
        self.setClusterEpsilon(clusterEpsilon)
        self.setHintEstimationThreshold(hintEstimationThreshold)
        self.setHintValidationThreshold(hintValidationThreshold)
        self.setEnableStats(enableStats)

    def setInitSampleAlgo(self, algo):
        self.initSampleAlgo = algo

    def setSampleBudget(self, num):
        self.sampleBudget = num

    def setNumForests(self, num):
        self.numForests = num

    def setNumTrees(self, num):
        self.numTrees = num

    def setRatioVarBootstrap(self, ratio):
        self.ratioVarBootstrap = ratio

    def setPdfThreshold(self, threshold):
        self.pdfThreshold = threshold
        
    def setNumDeltaThresholds(self, num):
        self.numDeltaThresholds = num

    def setElimAgreeThreshold(self, thresh):
        """
        Percentage of the estimators that agree that '(xp - x) > delta'
        """
        self.elimAgreeThreshold = thresh

    def setMinDeltaCandidates(self, num):
        self.minNumDeltaCandidates = num

    def setEliminationEnabled(self, enable):
        self.enableElimination = enable

    def setRandomSeed(self, seed):
        self.randomSeed = seed

    def setDataSaver(self, saver):
        self.dataSaver = saver

    def setNumWorkers(self, num):
        self.numWorkers = num
        if self.pool:
            self.pool.close()
        # Pool of processes, prevent them to receive KeyboardInterrupt
        self.pool = mp.Pool(self.numWorkers, lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))

    def setUseHintIfAvailable(self, useHint):
        self.useHint = useHint

    def setClusterRatioThreshold(self, threshold):
        self.clusterRatioThresh = threshold
        
    def setClusterBeta(self, beta):
        self.clusterBeta = beta
        
    def setClusterEpsilon(self, epsilon):
        self.clusterEpsilon = epsilon
        
    def setHintEstimationThreshold(self, num):
        """
        How many designs are estimated to be superior by the hint data
        for a design to be an elimination candidate.
        """
        self.hintEstimationThreshold = num

    def setHintValidationThreshold(self, num):
        """
        How many designs are estimated to be superior by the estimators
        for a hint elimination candidate to be validated.
        """
        self.hintValidationThreshold = num

    def setEnableStats(self, enable):
        self.enableStats = enable


    def run(self):
        if self.designs.getNumDesigns() <= 0: raise ValueError("No valid designs")
      
        nperr = np.seterr(invalid='ignore') # (ignore NaN)
      
        logger.info("Starting ATNE")

        np.random.seed(self.randomSeed)
        
        # Get knobs and replace missing values
        allKnobs   = self.designs.getKnobSettings()
        imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
        imp = imp.fit(allKnobs)
        allKnobs   = imp.transform(allKnobs)
        
        numDesigns = self.designs.getNumDesigns()
        alpha      = 1 - np.power(self.pdfThreshold, 1/np.arange(1,self.numDeltaThresholds+1))

        useHint = self.useHint and self.designs.hasHintsOnAllObjectives()
        logger.info(("U" if useHint else "Not u") + "sing hint elimination")

        # Define sets: L and P_relaxed
        # Note: L         == labeledIndexes
        #       P_relaxed == relaxedIndexes
        allIndexes     = np.arange(numDesigns, dtype=np.uint64)
        relaxedIndexes = allIndexes

            
        
        
        self.dataSaver.setDesigns(self.designs)
        self.dataSaver.setKnobs(allKnobs)
        self.dataSaver.setRelaxedIndexes(relaxedIndexes)
        self.dataSaver.setAlgorithmStart()


        ############################
        # Initial sampling
        ############################

        self.initSampleAlgo.setDesignData(self.designs)
        self.initSampleAlgo.run()
        initSampledIndexes = self.initSampleAlgo.getSampledIndexes()
        
        labeledIndexes = initSampledIndexes

        # Sample the initial designs
        labelsRaw    = self.designs.getGroundTruth(labeledIndexes)
        labeledKnobs = allKnobs[labeledIndexes]

        numKnobsPerSplit = labeledKnobs.shape[1]

        # Predict maximum values
        estimator = RandomForestRegressor(n_estimators=500, max_features='auto')
        estimator.fit(labeledKnobs, labelsRaw)
        estimatedMax = np.amax(estimator.predict(allKnobs), axis=0)
        logger.debug("Estimated max: " + str(estimatedMax))

        # Normalize
        labels = labelsRaw / estimatedMax

        numObjectives = labels.shape[1]

        self.kdeScores = np.array([]).reshape(0,numDesigns)

        self.dataSaver.setSampledIndexes(labeledIndexes)
        self.dataSaver.setIterationDone()

        self.stats = []
        adrs = np.nan

        # A bit of debug output
        debug_string = "Num Samples: " + str(labeledIndexes.size) + " ({:.2%})".format(labeledIndexes.size / numDesigns)

        if (self.enableStats or logger.isEnabledFor(logging.DEBUG)) and self.designs.hasGroundTruth():
            adrs = Distances.adrs(self.designs.getGroundTruth(range(self.designs.getNumDesigns())), labeledIndexes)
            debug_string += ", ADRS: " + str(adrs)

        logger.debug(debug_string)

        if self.enableStats:
            self.__recordStats(labeledIndexes.size, labeledIndexes.size / numDesigns, adrs)


        eliminationProbability = np.zeros((numDesigns,1,numObjectives))

        ############################
        # Active learning
        ############################

        while(labeledIndexes.size < self.sampleBudget                       # Stay within sampling budget
                and np.setdiff1d(relaxedIndexes, labeledIndexes).size > 0   # Do we have sampled everything we could?
                ):

            numLabeled = labeledIndexes.size
            numRelaxed = relaxedIndexes.size
            
            
            # Profiling
            timers = np.zeros(20)
            tidx = 0

            timers[tidx] = timer()
            tidx+=1
            

            ######################## Estimators ########################
            
            numBootstrapSamples = int(round(numLabeled * self.ratioVarBootstrap))


            # Regress using random forests, predict values:
            #    - Train on all available ground truth values ( function atne_createAndFitEstimators() )
            #    - Predict the P_relaxed set
            
            # Create and train estimators
            estimatorFuncArguments = [(numObjectives, numBootstrapSamples, numKnobsPerSplit, self.numTrees, self.ratioVarBootstrap, labeledKnobs, labels)] * self.numForests
            
            estimators = list(self.__mapFunction(atne_createAndFitEstimators, estimatorFuncArguments))
            
            
            # Predict
            predictions   = np.empty([self.numForests, numRelaxed, numObjectives])
            for f in range(self.numForests):
                for o in range(numObjectives):
                    predictions[f,:,o] = estimators[f][o].predict(allKnobs[relaxedIndexes])


            timers[tidx] = timer()
            tidx+=1

         
            ######################## Hint ########################
            hintEliminatedIndexes = np.empty(0,dtype=np.int)

            if useHint:
                
                # Get set to operate on
                hintIndexes = self.__getHintSet(relaxedIndexes, labeledIndexes)

                # Calculate distances within this set and cluster them
                (hintDistances, hintClusters, hintClustersRatios) = self.__getHintClusters(hintIndexes, labels, labeledIndexes)
                
                # From the clusters, get the designs to eliminate
                hintEliminatedIndexes = self.__getHintEliminatedIndexes(hintIndexes, hintDistances, hintClusters, hintClustersRatios)
                
                # Make sure we remove labeled designs from eliminated designs
                hintEliminatedIndexes = np.setdiff1d(np.intersect1d(relaxedIndexes, hintEliminatedIndexes, assume_unique=True),
                                                     labeledIndexes, assume_unique=True)
                
                finalHintEliminatedIndexes = []
                hintInRelaxedMask = np.in1d(relaxedIndexes, hintEliminatedIndexes, assume_unique=True)
                 
                # In the predicted space, verify that the eliminated designs are non-Pareto optimal
                for predIdx in np.where(hintInRelaxedMask)[0]:
                    distances = (predictions.swapaxes(0,1) - predictions[:,predIdx,:]).swapaxes(0,1)
                    np.all(distances > 0, 0)
                    # Do all the estimators agree on the distance sign, for all the objectives?
                    isInferior = np.all(np.all(distances > 0, 0), 1)
                     
                    # If x is inferior to a number of designs according to all estimators, then we validate the elimination
                    if np.sum(isInferior) >= self.hintValidationThreshold:
                        finalHintEliminatedIndexes.append(relaxedIndexes[predIdx])
                 
                hintEliminatedIndexes = np.array(finalHintEliminatedIndexes, dtype=np.int)
                
                logger.debug("Hint eliminated: " + str(hintEliminatedIndexes))

            timers[tidx] = timer()
            tidx+=1


            ######################## Estimate delta ########################
            
            # We work on the set of labeled designs that are in P_relaxed
            # Let Selected = (P_relaxed 'intersect' L)

            selectedIndexes = np.intersect1d(labeledIndexes, relaxedIndexes, assume_unique=True) # P_relaxed intersect L
            selectedMaskL   = np.in1d(labeledIndexes, selectedIndexes, assume_unique=True) # Boolean mask of selected designs within L
            selectedMaskP   = np.in1d(relaxedIndexes, selectedIndexes, assume_unique=True) # Boolean mask of selected designs within P_relaxed

            numSelectedSamples   = selectedIndexes.size
            numSelectedDistances = int((numSelectedSamples**2-numSelectedSamples)/2)

            delta              = np.zeros([self.numDeltaThresholds, numObjectives], dtype=float)
            numDeltaCandidates = np.zeros(numObjectives)

            # Compute a delta for each objective
            for o in range(numObjectives):
                
                # Compute distances for ground truth
                groundtruthDistances = pdist(labels[selectedMaskL,o].reshape([-1,1]), lambda x, xp: xp - x) # Slow
                
                # Compute corresponding estimated distances
                selectedDistances = np.empty([self.numForests, numSelectedDistances])
                for f in range(self.numForests):
                    selectedDistances[f,:] = pdist(predictions[f,selectedMaskP,o].reshape([-1,1]), lambda x, xp: xp - x) # Slow

                self.dataSaver.setPredictedDistancesInGroundTruthSet(selectedDistances, groundtruthDistances)
               
                # Mu / sigma: across forests, for each pair of points
                mu_x_xp    = selectedDistances.mean(0)
                sigma_x_xp = selectedDistances.std(0, ddof=1)

                deltaMask = (mu_x_xp * groundtruthDistances) < 0

                numDeltaCandidates[o] = np.sum(deltaMask)

                # Compute multiple deltas
                for i in range(self.numDeltaThresholds):
                    delta_x_xp = np.abs(mu_x_xp) + sigma_x_xp * np.sqrt(2) * erfinv(2*alpha[i]-1)

                    deltaCandidates = delta_x_xp[deltaMask]
                    delta[i, o]     = 0 if (deltaCandidates.size == 0 or alpha[i] <= 0) else max(deltaCandidates)

            for d in delta:
                if np.any(d < 0.5*delta[0]):
                    d.fill(np.inf)


            logger.debug("Num designs for delta: " + str(len(selectedIndexes)))
            logger.debug("Delta: \n" + str(delta))
            logger.debug("Num delta candidates: " + str(numDeltaCandidates))



            timers[tidx] = timer()
            tidx+=1


            ######################## Elimination ########################
            
            enoughDeltaToEliminate = np.all(numDeltaCandidates >= self.minNumDeltaCandidates) 

            eliminatedIndexes = np.array([], dtype=np.int)
            
            if enoughDeltaToEliminate and self.enableElimination:
       
                # This makes sure we don't check labeled designs
                # Basically we iterate on (P_relaxed - L)
                indexList = np.setdiff1d(np.arange(numRelaxed), np.where(np.in1d(relaxedIndexes, labeledIndexes))[0], assume_unique=True)
       
                # Note: DataSaver (and more specifically DataPlotter2D) is not thread-/process-safe
                elimFuncDataSaver = self.dataSaver if self.numWorkers == 1 else None
                   
                # For each x in indexList...
                elimFuncArguments = [(xIdx, self.numDeltaThresholds, numRelaxed, numObjectives, self.numForests, predictions, delta, relaxedIndexes, indexList, self.elimAgreeThreshold, elimFuncDataSaver)
                                for xIdx in indexList]
       
                # ... check if it should be eliminated 
                eliminatedIndexes = self.__mapFunction(atne_relaxedCheckEliminate, elimFuncArguments)
       
                # Remove all the "-1"
                eliminatedIndexes = np.array([e for e in eliminatedIndexes if e >= 0], dtype=np.int)
            


            # Find non Pareto-optimal designs within the ground truth data
            selectedOptimalMask = Distances.getParetoOptimalDesigns(labels[selectedMaskL,:])
            nonParetoOptimalIndexes = selectedIndexes[np.logical_not(selectedOptimalMask)]

            logger.debug("Eliminated: " + str(eliminatedIndexes))

            # New P_relaxed set: Remove eliminated designs and ground truth non-Pareto-optimal designs
            newRelaxedIndexes = np.setdiff1d(relaxedIndexes, eliminatedIndexes)
            newRelaxedIndexes = np.setdiff1d(newRelaxedIndexes, hintEliminatedIndexes)
            
            if enoughDeltaToEliminate:
                newRelaxedIndexes = np.setdiff1d(newRelaxedIndexes, nonParetoOptimalIndexes)
            
            newRelaxedMaskP   = np.in1d(relaxedIndexes, newRelaxedIndexes) # Boolean mask of the new P_relaxed set within the old P_relaxed


            timers[tidx] = timer()
            tidx+=1

            nextSampleIndexes = []

            ######################## Select next sample ########################

            # Calculate Pareto-optimal designs in new P_relaxed
            newRelaxedOptimalMask = Distances.getParetoOptimalDesigns(predictions[:,newRelaxedMaskP,:].mean(0))
            numRelaxedOptimal = np.sum(newRelaxedOptimalMask)

            # If we are close to the sample budget, select a design on the estimated Pareto front in P_relaxed
            if len(labeledIndexes) >= self.sampleBudget - numRelaxedOptimal:
                nextSampleIndexes = newRelaxedIndexes[newRelaxedOptimalMask]
                nextSampleIndexes = np.setdiff1d(nextSampleIndexes, labeledIndexes)
                if len(nextSampleIndexes) > 0:
                    nextSampleIndexes = np.random.choice(nextSampleIndexes, size=1, replace=False)


            # Make sure there are enough remaining unknown designs
            unknownIndexes = np.setdiff1d(newRelaxedIndexes, np.intersect1d(newRelaxedIndexes, labeledIndexes, assume_unique=True), assume_unique=True)
            unknownMaskP   = np.in1d(relaxedIndexes, unknownIndexes) # Boolean mask of unknown designs within P_relaxed
            unknownIdxToPIdx    = np.where(unknownMaskP)[0] # Translate from unknown set index to P_relaxed index
            numUnknownSamples   = unknownIndexes.size


            # If we haven't already selected the next sample, pick one based on a dominance score
            if len(nextSampleIndexes) == 0 and numUnknownSamples > 0:

                # From Matlab code
                # Calculate a score for each design:
                #    - Compute the difference between the design and all others
                #    - For each distance
                #         - if the distance is greater than -delta on all objectives, increase the score

                # For each x in unknown designs...
                selectFuncArguments = [(xIdx, predictions, unknownMaskP, delta) for i,xIdx in enumerate(unknownIdxToPIdx)]

                # ... calculate the selection score
                elimination_difficulty = np.fromiter(self.__mapFunction(atne_getSampleSelectionScore, selectFuncArguments), dtype=float)

                # Sample the design with the maximum score
                nextSampleIndexes = unknownIndexes[np.argmax(elimination_difficulty)]




            self.dataSaver.setPredictionsIndexes(relaxedIndexes)



            relaxedIndexes = newRelaxedIndexes


            # Make sure we are sampling only unlabeled designs
            nextSampleIndexes = np.setdiff1d(nextSampleIndexes, labeledIndexes)


            # Get the ground truth label for the next sample
            if len(nextSampleIndexes) > 0:
                nextSampleLabels = self.designs.getGroundTruth(nextSampleIndexes) / estimatedMax
    
                # Grow the set of labeled designs
                nextSampleInsertIdx = np.searchsorted(labeledIndexes, nextSampleIndexes)
    
                labeledIndexes = np.insert(labeledIndexes, nextSampleInsertIdx, nextSampleIndexes)
                labels         = np.insert(labels, nextSampleInsertIdx, nextSampleLabels, axis=0)

            labeledKnobs = allKnobs[labeledIndexes]


            debug_string = "Num Samples: " + str(labeledIndexes.size) + " ({:.2%})".format(labeledIndexes.size / numDesigns)

            if (self.enableStats or logger.isEnabledFor(logging.DEBUG)) and self.designs.hasGroundTruth():
                adrs = Distances.adrs(self.designs.getGroundTruth(range(self.designs.getNumDesigns())), labeledIndexes)
                debug_string += ", ADRS: " + str(adrs)

            logger.debug(debug_string)

            if self.enableStats:
                self.__recordStats(labeledIndexes.size, labeledIndexes.size / numDesigns, adrs)

            timers[tidx] = timer()
            tidx+=1
            
            self.dataSaver.setRelaxedIndexes(relaxedIndexes)
            self.dataSaver.setSampledIndexes(labeledIndexes)
            self.dataSaver.setHintEliminatedIndexes(np.setdiff1d(hintEliminatedIndexes, eliminatedIndexes, True))
            self.dataSaver.setPredictions(predictions, estimators)
            self.dataSaver.setIterationDone()
            

            timers[tidx] = timer()
            tidx+=1

            timers_diff = np.diff(timers)

            profiling = {}
            profiling['Random forest']    = (timers_diff[0]) * 1000
            profiling['Hint']             = (timers_diff[1]) * 1000
            profiling['Delta']            = (timers_diff[2]) * 1000
            profiling['Elimination']      = (timers_diff[3]) * 1000
            profiling['Sample selection'] = (timers_diff[4]) * 1000
            profiling['Save/Plot data']   = (timers_diff[5]) * 1000

            logger.debug(" ".join(map(str, sorted(profiling.items()))))

        ############
        # end while
        ############


        self.sampledIndexes = labeledIndexes
        self.stats = np.array(self.stats)
        
        self.dataSaver.setRelaxedIndexes(relaxedIndexes)
        self.dataSaver.setSampledIndexes(labeledIndexes)
        self.dataSaver.setAlgorithmDone()

        np.seterr(**nperr)

        logger.info("ATNE done")



    def __recordStats(self, numSamples, percentSamples, adrs):
        self.stats.append((numSamples, percentSamples, adrs))


    def __mapFunction(self, function, argumentList):
        if self.numWorkers == 1:
            result = map(function, argumentList)
        else:
            try:
                result = self.pool.map_async(function, argumentList).get(31536000)
            except KeyboardInterrupt:
                self.pool.terminate()
                self.pool.close()
                raise KeyboardInterrupt
        return result


    def __getHintSet(self, relaxedIndexes, labeledIndexes):
        """
        Get the set of designs on which we operate the clustering algorithms.
        """
        hintIndexes = np.union1d(relaxedIndexes, labeledIndexes)
            
        return hintIndexes


    def __getHintClusters(self, hintIndexes, labels, labeledIndexes):
        """
        Calculate the correlation between hint data and labeled data,
        then cluster the distances and correlations.
        """
        numHints      = hintIndexes.size
        hintInLabeled = np.in1d(hintIndexes, labeledIndexes, assume_unique=True) # Boolean mask of hint indexes that are labeled
        
        hints = self.designs.getHints(hintIndexes)
        
        numObjectives = labels.shape[1]
        numDistances  = int(numHints * (numHints-1) / 2)

        hintDistances     = np.zeros([numObjectives, numDistances]) # Distance matrix
        hintDistancesCorr = np.zeros([numObjectives, numDistances]) # [-1,0,1] matrix

        hintObjectiveIdx = 0

        # For each objective for which we do have hint information
        for o in range(numObjectives):
            if not self.designs.hasHints()[o]: continue

            # For each pair of designs in the hint set (python double loop: slow)
            distIdx = 0
            for hintI in range(numHints-1):
                for hintJ in range(hintI+1, numHints):

                    # Calculate difference
                    hintDist = hints[hintI, hintObjectiveIdx] - hints[hintJ, hintObjectiveIdx]

                    # If we have label for this pair, calculate labels distance
                    # and get a correlation coefficient (1 if same as hint, otherwise -1)
                    if hintInLabeled[hintI] and hintInLabeled[hintJ]:
                        labelsIdx = np.searchsorted(labeledIndexes, hintIndexes[[hintI,hintJ]])
                        labelsDistance = labels[labelsIdx[0],o] - labels[labelsIdx[1],o]

                        hintDistancesCorr[o, distIdx] = np.sign(labelsDistance * hintDist)

                    # Save hint distance
                    hintDistances[o, distIdx] = hintDist

                    distIdx += 1

            hintObjectiveIdx += 1


        clustersIdx = np.empty([numObjectives, numDistances], dtype=np.int)
        corrRatio   = []
        hintDistancesAbs = np.abs(hintDistances)
        
        for o in range(numObjectives):
            permutations         = np.argsort(hintDistancesAbs[o,:])
            inversePermutations  = np.empty(len(permutations), dtype=np.int)
            inversePermutations[permutations] = np.arange(len(permutations))

            (clusters, ratio) = self.__clusterHintDistances(hintDistancesAbs[o,:][permutations], hintDistancesCorr[o,:][permutations])

            clustersIdx[o,:] = clusters[inversePermutations]
            corrRatio.append(ratio)

        return (hintDistances, clustersIdx, corrRatio)


    def __clusterHintDistances(self, hintDistances, hintDistancesCorr):
        """
        Cluster the given hint distances. The distances must be sorted!
        This works for one objective (i.e., the given arrays must be 1-dimensional).
        This is a recursive method.
        """
        N = hintDistances.size

        if N == 0:
            return (np.empty(dtype=np.int), np.empty())

        # Calculate the threshold Nth (size of clusters)
        K = round(N*(1-self.clusterEpsilon))
        for x in range(1, N+1):
            if (comb(K, x) / comb(N, x)) <= self.clusterBeta: break
        Nth = x

        numSamples = np.count_nonzero(hintDistancesCorr)
        numOnes    = np.count_nonzero(hintDistancesCorr == 1)

        # Compute the homogeneity of the correlations on the current cluster
        ratio = 0.5 # 0.5 means we don't have information on this cluster
        if numSamples != 0:
            ratio = numOnes / numSamples
        corrRatio = np.array([[1-ratio, ratio]])
        
        clusters  = np.empty(N, dtype=np.int)

        allDistancesEqual = (hintDistances[0] == hintDistances[-1])

        # If cluster is larger than threshold OR the cluster is not pure enough, split
        if numSamples > Nth and np.max(corrRatio) < self.clusterRatioThresh and not allDistancesEqual:

            #### Commented below are various methods
            #### Mostly they use up a lot of memory, except for myClusters
            # Find 2 clusters
            #clustersIdx = fclusterdata(hintDistances.reshape(-1,1), 2, criterion='maxclust', method='ward')
            #clustersIdx = AgglomerativeClustering().fit(hintDistances.reshape(-1,1)).labels_
            #clustersIdx = self.__myClusters(hintDistances)

            # Recursively cluster the 2 clusters (see below to work on already sorted data)
            #(clustersIdx1, corrRatio1) = self.__clusterHintDistances(hintDistances[clustersIdx==1], hintDistancesCorr[clustersIdx==1], beta, epsilon)
            #(clustersIdx2, corrRatio2) = self.__clusterHintDistances(hintDistances[clustersIdx==2], hintDistancesCorr[clustersIdx==2], beta, epsilon)


            # Simple clustering for already sorted data
            splitIdx = np.argmax(np.diff(hintDistances)) + 1

            # Recursively cluster the 2 clusters
            (clustersIdx1, corrRatio1) = self.__clusterHintDistances(hintDistances[:splitIdx], hintDistancesCorr[:splitIdx])
            (clustersIdx2, corrRatio2) = self.__clusterHintDistances(hintDistances[splitIdx:], hintDistancesCorr[splitIdx:])


            # Merge the left and right clusters
            if clustersIdx1.size != 0:
                clustersIdx2 += (clustersIdx1[-1] + clustersIdx1.dtype.type(1))

            #clusters[clustersIdx==1] = clustersIdx1
            #clusters[clustersIdx==2] = clustersIdx2
            
            clusters[:splitIdx] = clustersIdx1
            clusters[splitIdx:] = clustersIdx2
            
            corrRatio = np.vstack((corrRatio1, corrRatio2)) 

        # Otherwise stop the algorithm, we have a cluster
        else:
            clusters.fill(0)
            # We remove information about this cluster if number of samples is too small
            if numSamples < Nth:
                corrRatio[:] = 0.5

        return (clusters, corrRatio)


    def __myClusters(self, data):
        """
        Find the largest gap in sorted data.
        Return the clusters (1,2) for the original data.
        """
        clusterIdx  = np.ones(data.size, dtype=np.uint8)
        dataSortIdx = np.argsort(data.flatten())
        dataSorted  = data.flatten()[dataSortIdx]
        splitIdx    = np.argmax(np.diff(dataSorted))

        clusterIdx[dataSortIdx[:splitIdx+1]] = np.uint8(2)

        return clusterIdx



    def __getHintEliminatedIndexes(self, hintIndexes, hintDistances, hintClusters, hintClustersRatios):

        numHints      = hintIndexes.size
        numObjectives = len(hintClustersRatios)
        numDistances  = hintDistances.shape[1]

        estimatedNonPareto = np.zeros(numHints, dtype=np.int)

        # Translate from distance index to design indexes (i,j)
        def idxToIJ(idx):
            b = 1 - 2*numHints
            i = int(np.floor((-b - np.sqrt(b**2 - 8*idx))/2))
            j = int(idx + i*(b + i + 2)/2 + 1)
            return (i,j)
        
        # Opposite function
        # ijToIdx = lambda i,j: numHints*i - i*(i+1)/2 + j - i - 1


        # Pad with dummy values so we can work with a pure numpy array
        maxNumClusters = np.max([x.shape[0] for x in hintClustersRatios])
        hintClustersRatios = np.array([np.pad(x, ((0,maxNumClusters-len(x)),(0,0)), 'constant', constant_values=(0.5,0.5)) for x in hintClustersRatios])
        
        
        # For each pair of designs in the hint set, calculate the estimated distance sign
        # to determine if the design is non Pareto optimal

        if self.numWorkers == 1:
            # Algorithm for single-threaded program
            # Note: It is faster to write the entire algorithm than re-using the 'atne_hintEstimatedDistanceInCluster' function
            
            # For each distance
            for distIdx in range(numDistances):
     
                distCluster = hintClusters[:,distIdx]
            
                distClusterRatio = hintClustersRatios[np.arange(numObjectives), distCluster]
                
                # If this distance belongs to a cluster that is "pure" enough
                if np.all(np.max(distClusterRatio,1) >= self.clusterRatioThresh):
            
                    # Get the correlation "direction" (i.e. the cluster is mostly composed of (1) or (-1))
                    correlationDirection = np.ones(numObjectives)
                    correlationDirection[distClusterRatio[:,0] > distClusterRatio[:,1]] = -1
            
                    estimatedDistance = correlationDirection * hintDistances[:,distIdx]
                    
                    if np.all(estimatedDistance < 0):
                        # Design j is superior to i
                        hintI = idxToIJ(distIdx)[0]
                        estimatedNonPareto[hintI] += 1
                    elif np.all(estimatedDistance > 0):
                        # Design i is superior to j
                        hintJ = idxToIJ(distIdx)[1]
                        estimatedNonPareto[hintJ] += 1
            
        else:
            # Same algo for multiple threads
            try:
                arguments = [(distIdx, hintDistances, hintClusters, hintClustersRatios, numObjectives, self.clusterRatioThresh)
                             for distIdx in range(numDistances)]
                
                estimatedDistanceSigns = self.pool.map_async(atne_hintEstimatedDistanceInCluster, arguments).get(31536000)
                
                for idx in np.where(estimatedDistanceSigns)[0]:
                    dist = estimatedDistanceSigns[idx]
                    if dist < 0:
                        # Design j is superior to i
                        hintI = idxToIJ(idx)[0]
                        estimatedNonPareto[hintI] += 1 
                    elif dist > 0:
                        # Design i is superior to j
                        hintJ = idxToIJ(idx)[1]
                        estimatedNonPareto[hintJ] += 1
    
            except KeyboardInterrupt:
                self.pool.terminate()
                self.pool.close()
                raise KeyboardInterrupt


        hintEliminatedIndexes = hintIndexes[estimatedNonPareto >= self.hintEstimationThreshold]

        return hintEliminatedIndexes
        


class RandomSampler(DesignSampler):
    def __init__(self,
            designs = DesignDataIO.DesignDataBase(),
            numSamples = 20,
            randomSeed = None):
        super(RandomSampler, self).__init__(designs)
        self.setNumSamples(numSamples)
        self.setRandomSeed(randomSeed)

    def setNumSamples(self, numSamples):
        self.numSamples = numSamples

    def setRandomSeed(self, seed):
        self.randomSeed = seed

    def run(self):
        rand_gen = np.random.RandomState(self.randomSeed)

        self.sampledIndexes = rand_gen.choice(
                range(self.designs.getNumDesigns()),
                self.numSamples,
                replace=False)


