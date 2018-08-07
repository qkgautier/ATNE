#!/usr/bin/env python

"""
License and copyright TBD soon

Author: Quentin Gautier
"""

from __future__ import division

import logging
import argparse
import os
import numpy as np
import multiprocessing

import DesignDataIO
import DesignSampling
import Distances



def main(clArgs=None):

    tedTypeList = ["ted", "half_hint", "hint_output", "hint_all_output", "pareto", "pareto_inv"]
    
    ####### Parameters #######

    parser = argparse.ArgumentParser(description="Run the ATNE algorithm")

    atne_group = parser.add_argument_group("ATNE")
    atne_group.add_argument("-i",    "--input-file",       default='../data/bfs_sparse.mat', help='Path to the input file (default: %(default)s)')
    atne_group.add_argument("-nis",  "--num-init-samples", default = 15,   type=int,   help='Number of initial samples (default: %(default)s)')
    atne_group.add_argument("-sb",   "--sample-budget",    default = 1.00,  type=float, help='Total number of designs to sample (default: %(default)s) (If in [0.0 .. 1.0], percentage of design candidates)')
    atne_group.add_argument("-nf",   "--num-forests",      default = 30,    type=int,   help='Number of forests (default: %(default)s)')
    atne_group.add_argument("-nt",   "--num-trees",        default = 200,   type=int,   help='Number of trees (default: %(default)s)')
    atne_group.add_argument("-rb",   "--ratio-bootstrap",  default = 0.50, type=float, help='Bootstrap size in percentage of the set size (default: %(default)s)')
    atne_group.add_argument("-pdf",  "--pdf-threshold",    default = 0.001, type=float, help='PDF threshold for delta calculation (default: %(default)s)')

    atne_group.add_argument("-de",   "--disable-elim",                                 help='Disable the elimination step (default: %(default)s)', action='store_true')
    atne_group.add_argument("-rs",   "--random-seed",      default = None, type=int,   help='Random seed (default: %(default)s)')
    atne_group.add_argument("-rand", "--random-sampling",                              help='Use random sampling instead of ATNE (only considers the sample budget option) (default: %(default)s)', action='store_true')
    atne_group.add_argument("-ted",  "--ted-only",                                     help='Use TED instead of ATNE (only considers the sample budget option) (default: %(default)s)', action='store_true')
    
    hint_group = parser.add_argument_group("ATNE hint")
    hint_group.add_argument("-hint", "--hint",                                   help='Enable hint elimination (default: %(default)s)', action='store_true')
    hint_group.add_argument("-tedt", "--ted-type",        default = "ted",       help="Which TED method: " + str(tedTypeList) + ". (default: %(default)s)")
    hint_group.add_argument("-cb",   "--cluster-beta",    default = 0.1, type=float, help="Beta parameter to calculate cluster size (default: %(default)s)")
    hint_group.add_argument("-ce",   "--cluster-epsilon", default = 0.25, type=float, help="Epsilon parameter to calculate cluster size (default: %(default)s)")
#     hint_group.add_argument("-nhe", "--no-hint-elim",                             help="Disable hint elimination. (default: %(default)s)", action='store_true')
    hint_group.add_argument("-ht",   "--hint-type",       default = "gpu",            help="Type of Hint (gpu, esttp, cpu). (default: %(default)s)")
    atne_group.add_argument("-hpar", "--hint-pareto",     default = -1.0, type=float, help='Sample designs that are Pareto-optimal (with margin) on hint space only (disable ATNE). Enabled if margin >= 0. (default: %(default)s)')
    
    plot_group = parser.add_argument_group("Plotting")
    plot_group.add_argument("-hl", "--headless",          help='Disable plotting (default: %(default)s)', action='store_true')
    plot_group.add_argument("-pb", "--plot-blocking",     help='Makes plot blocking (default: %(default)s)', action='store_true')
    plot_group.add_argument("-pd", "--plot-distances",    help='Enable distances plot (default: %(default)s)', action='store_true')
    plot_group.add_argument("-dp", "--disable-plot-pred", help='Disable forest predictions plot (default: %(default)s)', action='store_true')
    plot_group.add_argument("-dh", "--disable-plot-hint", help='Disable hint plot (will be disabled if not using hints) (default: %(default)s)', action='store_true')
    
    misc_group = parser.add_argument_group("Other options")
    misc_group.add_argument(       "--log",           default="debug", help='Logging level (debug, info, warning) (default: %(default)s)')
    misc_group.add_argument("-np", "--num-processes", default = None, type=int, help='Number of parallel processes (default: max available processes)')
    misc_group.add_argument("-nr", "--num-runs",      default = 1, type=int, help='Number of times to run the algorithm (default: %(default)s)')
    misc_group.add_argument("-o",  "--output",        help='Output the final results into a CSV file')
    misc_group.add_argument("-os",  "--output-stats", help='Output average stats into a NPZ file')
    misc_group.add_argument(       "--indexes",       help="Only keep the given indexes in the data (idx1,id2,idx3,...)")

    args = parser.parse_args(clArgs)


    filename = args.input_file
    filetype = os.path.splitext(filename)[1][1:]
    numInitSamples    = args.num_init_samples
    sampleBudget      = args.sample_budget
    numForests        = args.num_forests
    numTrees          = args.num_trees
    ratioVarBootstrap = args.ratio_bootstrap
    pdfThreshold      = args.pdf_threshold
    enableElimination = not args.disable_elim
    randomSeed        = args.random_seed
    useRandomSampling = args.random_sampling
    useTedOnly        = args.ted_only 
    
    useHint           = args.hint
    hintType          = args.hint_type
    tedType           = args.ted_type
    hintParetoMargin  = args.hint_pareto
    hintParetoOnly    = (hintParetoMargin >= 0)

    headless        = args.headless
    plotBlocking    = args.plot_blocking
    plotDistances   = args.plot_distances
    plotPredictions = not args.disable_plot_pred
    plotHintSpace   = not args.disable_plot_hint and useHint

    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log)
    logging.basicConfig(level=numeric_level, format="%(levelname)s:%(name)s: %(message)s")

    numWorkers  = multiprocessing.cpu_count() if args.num_processes is None else args.num_processes
    numRuns     = args.num_runs
    output      = args.output
    outputStats = args.output_stats
    
    if useRandomSampling or useTedOnly or hintParetoOnly:
        headless = True
        
    if useRandomSampling + useTedOnly + hintParetoOnly >= 2:
        raise Exception("You selected multiple options that disable ATNE. You can only select one.")
    
    try:
        tedType = tedTypeList[int(tedType)]
    except ValueError:
        if tedType not in tedTypeList:
            raise Exception("TED type not regognized")
    
    
    
    ####### Setup algorithm #######

    # Load data
    if filetype == "mat":
        designs = DesignDataIO.MatlabDesignData(filename, hintType=hintType)
    elif filetype == "csv":
        designs = DesignDataIO.CsvDesignData(filename, hint=['hint_time', 'hint_logic'])
    else:
        raise Exception("File extension not recognized: " + filetype)
    

    
    
    
    
    if args.indexes:
        designs.setIndexesToKeep(list(map(int, args.indexes.split(","))))
    
    numDesigns = designs.getNumDesigns()

    logging.info(str(numDesigns) + " designs loaded (hints: " + str(designs.hasHints()) + ")")


    if sampleBudget <= 1.0:
        sampleBudget = sampleBudget * designs.getNumDesigns()
    sampleBudget = int(sampleBudget)

    plotter = DesignDataIO.DataSaver()

    if not headless:
        plotter = DesignDataIO.DataPlotter2D(
                blocking        = plotBlocking,
                plotPredictions = plotPredictions,
                plotDistances   = plotDistances,
                plotHintSpace   = plotHintSpace)


    if useRandomSampling:
        sampler = DesignSampling.RandomSampler(designs    = designs,
                                               numSamples = sampleBudget)
    elif useTedOnly:
        sampler = DesignSampling.TED(designs    = designs,
                                     numSamples = sampleBudget,
                                     method = tedTypeList.index(tedType),
                                     enableStats = (outputStats != ""))
    elif hintParetoOnly:
        sampler = DesignSampling.HintParetoSampler(designs = designs, margin = hintParetoMargin)
        
    else:
        if tedType == "pareto": # DEBUG for now
            ted = DesignSampling.GroundTruthParetoSampler()
        elif tedType == "pareto_inv":
            ted = DesignSampling.GroundTruthParetoSampler(inverse=True)
        else:
            ted = DesignSampling.TED(numSamples=numInitSamples,
                                     method=tedTypeList.index(tedType))
    
        sampler = DesignSampling.ATNE(
            designs            = designs,
            initSampleAlgo     = ted,
            sampleBudget       = sampleBudget,
            numForests         = numForests,
            numTrees           = numTrees,
            ratioVarBootstrap  = ratioVarBootstrap,
            pdfThreshold       = pdfThreshold,
            enableElimination  = enableElimination,
            randomSeed         = randomSeed,
            dataSaver          = plotter,
            useHintIfAvailable = useHint,
            numWorkers         = numWorkers,
            enableStats        = (outputStats != "")
            )

    # Prepare CSV output

    if output:
        outFile = open(output, 'wt')

    parametersHeader = "numDesigns,numInitSamples,sampleBudget,numForests,numTrees,ratioVarBootstrap,pdfThreshold,enableElimination,randomSeed,useHint,hintEstThroughput,hintElimination,hintType,tedType,tedOnly,hintParetoOnly,randomSampling"
    
    parametersCsv = (str(numDesigns) + "," + str(numInitSamples) + "," + str(sampleBudget) + "," + str(numForests) + "," + str(numTrees) + ","
                     + str(ratioVarBootstrap) + "," + str(pdfThreshold) + ","
                     + str(enableElimination) + "," + str(randomSeed) + "," + str(useHint) + ","
                     + str(hintType=="esttp") + ","
                     + str(useHint) + "," + str(hintType) + "," + str(tedType) + ","
                     + str(useTedOnly) + "," + str(hintParetoOnly) + "," + str(useRandomSampling))



    ####### Run algorithm #######
    
    adrsResults     = []
    mdrsResults     = []
    selectedDesigns = []
    stats           = []
    
    for i in range(numRuns):
        
        if not headless:
            plotter.reset()

        sampler.run()
        
        # Calculate results
        finalDesigns = sampler.getSampledIndexes()
        adrs, mdrs = Distances.adrs_mdrs(designs.getGroundTruth(), finalDesigns)
        
        # Store results
        adrsResults.append(adrs)
        mdrsResults.append(mdrs)
        selectedDesigns.append(finalDesigns)

        if outputStats:
            stats.append(sampler.stats)

        if output:
            resultsCsv = ",".join([str(len(finalDesigns)), str(adrs), str(mdrs)])
            if i == 0:
                outFile.write("num_samples,adrs,mdrs," + parametersHeader + "\n")
                outFile.write(resultsCsv + "," + parametersCsv + "\n")
            else:
                outFile.write(resultsCsv + "\n")
            outFile.flush()


    ####### Print results #######    

    numSamplesStr = list(map(lambda x: str(len(x)), selectedDesigns))
    numSamplesCsv = ",".join(numSamplesStr)
    
    adrsStr       = list(map(str, adrsResults))
    adrsCsv       = ",".join(adrsStr)
    
    mdrsStr     = list(map(str, mdrsResults))
    mdrsCsv     = ",".join(mdrsStr)


    print("----------------------------------")
    print(parametersHeader)
    print(parametersCsv)
    print("Num selected samples: " + numSamplesCsv)
    print("ADRS = " + adrsCsv)
    print("MDRS = " + mdrsCsv)
    print("----------------------------------")

    if outputStats:
        if outputStats[-4:].lower() != '.npz':
            outputStats += '.npz'
        np.savez(outputStats, *stats)
        

    if not headless:
        plotter.block()


    return adrsResults, mdrsResults, selectedDesigns



if __name__ == "__main__":
    main()



