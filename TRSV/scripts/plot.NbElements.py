#!/usr/bin/env python
import Gnuplot
import math
import sys

def readDataFromFile(filename):
    f = open(filename, 'r')

    results = []
    for line in f:
        if line.find("Throughput") != -1: 
            results.append([line.split()[10][0:len(line.split()[10]) - 1], line.split()[2]])

    f.close()  
    return results

def readDataFromFileAll(filename, str1, str2):
    f = open(filename, 'r')

    results = []
    for line in f:
        if line.find("Time = ") != -1 and line.find(str1) != -1 and line.find(str2) != -1:
            line = line.split();
            #results.append([line[11], line[17]])
            results.append([line[11], line[14]])

    f.close()
    return results

def plotNbElementsVSGbs(input, output):
    results = readDataFromFile(input)

    # plot the results
    gp = Gnuplot.Gnuplot(persist=1)
    gp('set terminal postscript eps enhanced "Times" 26')
    gp('set grid noxtics ytics')
    gp('set xlabel "Array Size" font "Times, 26"')
    gp('set ylabel "Gb/s" font "Times, 26"')
    gp('set label "Expansion 2 + accumulator" textcolor lt -1 at first 50.2, first 11')
    gp('set yrange [0:]')
    gp('set output "| epstopdf --filter > ' + output + '"')    

    # legend
    gp('set key width -3.5 samplen 1.8')
    gp('set key top left')

    # margins
    gp('set tmargin .5')
    gp('set rmargin .5')
    gp('set lmargin 5.7')

    plot1 = Gnuplot.Data(results, with_='linespoints lt 1 lw 3.0 lc rgb "black" pt 2', title=None)

    gp.plot(plot1)
    return

def plotNbElementsVSGbsAll(input, output): 
    naive = readDataFromFileAll(input, "Alg = 0", "Alg = 0")
    print naive
    superaccs = readDataFromFileAll(input, "Alg = 1", "Alg = 1")
    fpe2 = readDataFromFileAll(input, "Alg = 2", "NbFPE = 2")
    fpe3 = readDataFromFileAll(input, "Alg = 2", "NbFPE = 3")
    fpe4 = readDataFromFileAll(input, "Alg = 2", "NbFPE = 4")
    fpe5 = readDataFromFileAll(input, "Alg = 2", "NbFPE = 5")
    fpe6 = readDataFromFileAll(input, "Alg = 2", "NbFPE = 6")
    fpe7 = readDataFromFileAll(input, "Alg = 2", "NbFPE = 7")
    fpe8 = readDataFromFileAll(input, "Alg = 2", "NbFPE = 8")
    fpe4ee = readDataFromFileAll(input, "Alg = 3", "NbFPE = 4")
    fpe6ee = readDataFromFileAll(input, "Alg = 4", "NbFPE = 6")
    print fpe6ee
    fpe8ee = readDataFromFileAll(input, "Alg = 5", "NbFPE = 8")

    # plot the results
    gp = Gnuplot.Gnuplot(persist=1)
    gp('set terminal postscript eps color enhanced "Times" 26')
    gp('set grid noxtics noytics')
    gp('set xlabel "Matrix size [n]" font "Times, 26"')
    gp('set ylabel "Time [secs]" font "Times, 26"')
    #gp('set ylabel "Time [secs]" font "Times, 26"')
    gp('set xrange [0:17000]')
    #gp('set logscale x 10')
    #gp('set logscale y')
    gp('set yrange [0:]')
    gp('set output "| epstopdf --filter > ' + output + '"')

    # legend
    gp('set key width -3.5 samplen 1.8')
    gp('set key top left')

    # margins
    gp('set tmargin .5')
    gp('set rmargin 1.')
    gp('set lmargin 7.5')

    plotNaive = Gnuplot.Data(naive, with_='lines lt 1 lw 4.0', title="Parallel DTRSV")
    plotSuperaccs = Gnuplot.Data(superaccs, with_='lines lt 7 lw 4.0', title="Superacc")
    plotfpe2 = Gnuplot.Data(fpe2, with_='lines lt 8 lw 4.0', title="Expansion 2")
    plotfpe3 = Gnuplot.Data(fpe3, with_='lines lt 5 lw 4.0', title="FPE3 + Superacc")
    plotfpe4 = Gnuplot.Data(fpe4, with_='lines lt 2 lw 4.0', title="FPE4 + Superacc")
    plotfpe5 = Gnuplot.Data(fpe5, with_='lines lt 6 lw 4.0', title="Expansion 5")
    plotfpe6 = Gnuplot.Data(fpe6, with_='lines lt 8 lw 4.0', title="FPE6 + Superacc")
    plotfpe7 = Gnuplot.Data(fpe7, with_='lines lt 0 lw 4.0', title="Expansion 7")
    plotfpe8 = Gnuplot.Data(fpe8, with_='lines lt 4 lw 4.0', title="FPE8 + Superacc")
    plotfpe4ee = Gnuplot.Data(fpe4ee, with_='lines lt 0 lw 4.0', title="Expansion 4 early-exit")
    plotfpe6ee = Gnuplot.Data(fpe6ee, with_='lines lt 9 lw 4.0', title="FPE6EE + Superacc")
    plotfpe8ee = Gnuplot.Data(fpe8ee, with_='lines lt 3 lw 4.0', title="Expansion 8 early-exit")

    #gp.plot(plotNaive, plotSuperaccs, plotfpe2, plotfpe3, plotfpe4, plotfpe5, plotfpe6, plotfpe7, plotfpe8, plotfpe8ee)
    gp.plot(plotNaive, plotSuperaccs, plotfpe3, plotfpe4, plotfpe6, plotfpe8, plotfpe6ee)
    return

#plotNbElementsVSGbs(sys.argv[1], sys.argv[2])
plotNbElementsVSGbsAll(sys.argv[1], sys.argv[2])

