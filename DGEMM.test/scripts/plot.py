#!/usr/bin/env python
import Gnuplot
import math
import sys

def readDataFromFileAll(filename, str1, str2):
    f = open(filename, 'r')

    results = []
    for line in f:
        if line.find("Time = ") != -1 and line.find(str1) != -1 and line.find(str2) != -1:
            line = line.split();
            results.append([line[11], float(line[14])])

    f.close()
    return results

def readDataFromFileAllOld(filename, str1, str2):
    f = open(filename, 'r')

    results = []
    for line in f:
        if line.find("Performance = ") != -1 and line.find(str1) != -1 and line.find(str2) != -1:
            if line.find("NbFPE") != -1:
                line = line.split();
                results.append([line[11], float(line[len(line) - 2])])
            else:
                line = line.split();
                results.append([line[11], float(line[len(line) - 2])])

    f.close()
    return results

def plotNbElementsVSGbsAll(input, output): 
    mine = readDataFromFileAll(input, "Alg = 0", "Alg = 0")
    #amd = readDataFromFileAll(input, "Alg = 1", "Alg = 1")
    nvidia = readDataFromFileAll(input, "Alg = 2", "Alg = 2")
    sapr = readDataFromFileAll(input, "Alg = 30", "Alg = 30")
    fpepr3 = readDataFromFileAll(input, "Alg = 31", "NbFPE = 3")
    fpepr4 = readDataFromFileAll(input, "Alg = 31", "NbFPE = 4")
    fpepr5 = readDataFromFileAll(input, "Alg = 31", "NbFPE = 5")
    fpepr6 = readDataFromFileAll(input, "Alg = 31", "NbFPE = 6")
    fpepr7 = readDataFromFileAll(input, "Alg = 31", "NbFPE = 7")
    fpepr8 = readDataFromFileAll(input, "Alg = 31", "NbFPE = 8")
    fpeexpr4 = readDataFromFileAll(input, "Alg = 32", "Alg = 32")
    fpeexpr6 = readDataFromFileAll(input, "Alg = 33", "Alg = 33")
    sagl = readDataFromFileAll(input, "Alg = 50", "Alg = 50")
    fpegl3 = readDataFromFileAll(input, "Alg = 51", "NbFPE = 3")
    fpegl4 = readDataFromFileAll(input, "Alg = 51", "NbFPE = 4")
    fpegl5 = readDataFromFileAll(input, "Alg = 51", "NbFPE = 5")
    fpegl6 = readDataFromFileAll(input, "Alg = 51", "NbFPE = 6")
    fpegl7 = readDataFromFileAll(input, "Alg = 51", "NbFPE = 7")
    fpegl8 = readDataFromFileAll(input, "Alg = 51", "NbFPE = 8")
    fpeexgl4 = readDataFromFileAll(input, "Alg = 52", "Alg = 52")
    fpeexgl8 = readDataFromFileAll(input, "Alg = 53", "Alg = 53")

    # plot the results
    gp = Gnuplot.Gnuplot(persist=1)
    gp('set terminal postscript eps color enhanced "Times" 26')
    #gp('set terminal epslatex color colortext')
    gp('set grid noxtics noytics')
    gp('set xlabel "Matrix size [m = n = k]" font "Times, 26"')
    #gp('set xlabel "Matrix size [m = n = k]"')
    gp('set ylabel "Time [secs]" font "Times, 26"')
    #gp('set ylabel "Time [secs]"')
    gp('set xrange [0:2200]')
    gp('set yrange [0:2]')
    #gp('set output "| epstopdf --filter > ' + output + '"')
    gp('set output "' + output + '"')

    # legend
    gp('set key width -7.0 samplen 1.8')
    gp('set key top left')

    # margins
    gp('set tmargin .5')
    gp('set rmargin 2.5')
    gp('set lmargin 8.')

    #plotmine = Gnuplot.Data(mine, with_='lines lt 3 lw 4.0', title="Mine DGEMM")
    #plotamd = Gnuplot.Data(amd, with_='lines lt 1 lw 4.0', title="AMD DGEMM")
    plotnvidia = Gnuplot.Data(nvidia, with_='lines lt 1 lw 4.0', title="DGEMM")
    plotsa = Gnuplot.Data(sapr, with_='lines lt 7 lw 4.0', title="Superaccumulator")
    plotfpe3 = Gnuplot.Data(fpepr3, with_='lines lt 5 lw 4.0', title="Expansion 3")
    plotfpe4 = Gnuplot.Data(fpepr4, with_='lines lt 2 lw 4.0', title="Expansion 4")
    #plotfpe5 = Gnuplot.Data(fpepr5, with_='lines lt 6 lw 4.0', title="FPE 5")
    plotfpe6 = Gnuplot.Data(fpepr6, with_='lines lt 0 lw 4.0', title="Expansion 6")
    #plotfpe7 = Gnuplot.Data(fpepr7, with_='lines lt 0 lw 4.0', title="FPE 7")
    plotfpe8 = Gnuplot.Data(fpepr8, with_='lines lt 4 lw 4.0', title="Expansion 8")
    plotfpe4ex = Gnuplot.Data(fpeexpr4, with_='lines lt 3 lw 4.0', title="Expansion 4 early-exit")
    #plotfpe6ex = Gnuplot.Data(fpeexpr6, with_='lines lt 10 lw 4.0', title="FPE 6 EX")
    #plotsa = Gnuplot.Data(sagl, with_='lines lt 1 lw 4.0', title="Superacc")
    #plotfpe3 = Gnuplot.Data(fpegl3, with_='lines lt 5 lw 4.0', title="FPEGL 3")
    #plotfpe4 = Gnuplot.Data(fpegl4, with_='lines lt 2 lw 4.0', title="FPEGL 4")
    #plotfpe5 = Gnuplot.Data(fpegl5, with_='lines lt 6 lw 4.0', title="FPEGL 5")
    #plotfpe6 = Gnuplot.Data(fpegl6, with_='lines lt 7 lw 4.0', title="FPEGL 6")
    #plotfpe7 = Gnuplot.Data(fpegl7, with_='lines lt 0 lw 4.0', title="FPEGL 7")
    #plotfpe8 = Gnuplot.Data(fpegl8, with_='lines lt 4 lw 4.0', title="FPEGL 8")
    #plotfpe4ex = Gnuplot.Data(fpeexgl4, with_='lines lt 9 lw 4.0', title="FPEEXGL 4")
    #plotfpe8ex = Gnuplot.Data(fpeexgl8, with_='lines lt 3 lw 4.0', title="FPEEXGL 8")

    gp.plot(plotnvidia, plotsa, plotfpe3, plotfpe4, plotfpe6, plotfpe8, plotfpe4ex)
    #gp.plot(plotmine, plotnvidia, plotsa, plotfpe3, plotfpe4, plotfpe5, plotfpe6, plotfpe7, plotfpe8, plotfpe4ex)
    #gp.plot(plotsa, plotfpe2, plotfpe3, plotfpe4, plotfpe5, plotfpe6, plotfpe7, plotfpe8, plotfpe4ex, plotfpe8ex)
    return

plotNbElementsVSGbsAll(sys.argv[1], sys.argv[2])

