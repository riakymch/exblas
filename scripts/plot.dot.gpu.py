#!/usr/bin/env python
import Gnuplot
import math
import sys

def plotNbElementsVSGbsAll(input1, input2, output): 
    dot = []
    #readDataFromFileAll(input1, "Alg = 0", "Alg = 0")
    superacc = []
    fpe3 = []
    fpe4 = []
    fpe8 = []
    fpe4ee = []
    fpe6ee = []
    fpe8ee = []
    f2 = open(input1, 'r')
    for line in f2:
        if line.find("Alg = 0") != -1:
            line = line.split()
            dot.append([line[8], float(line[14])])
    print dot
    f2.close()
    f = open(input2, 'r')
    for line in f:
        line = line.split()
        '''mult = 1e-9*float(line[0])
        superacc.append([line[0], mult / float(line[2])]);
        fpe2.append([line[0], mult / float(line[3])]);
        fpe3.append([line[0], mult / float(line[4])]);
        fpe4.append([line[0], mult / float(line[5])]);
        fpe8.append([line[0], mult / float(line[6])]);
        fpe8ee.append([line[0], mult / float(line[7])]);'''
        superacc.append([line[0], float(line[2])]);
        fpe3.append([line[0], float(line[3])]);
        fpe4.append([line[0], float(line[4])]);
        fpe8.append([line[0], float(line[5])]);
        fpe4ee.append([line[0], float(line[6])]);
        fpe6ee.append([line[0], float(line[7])]);
        fpe8ee.append([line[0], float(line[8])]);
    f.close()

    # plot the results
    gp = Gnuplot.Gnuplot(persist=1)
    gp('set terminal postscript eps color enhanced "Times" 26')
    gp('set grid noxtics noytics')
    gp('set xlabel "Array size" font "Times, 26"')
    #gp('set ylabel "Gacc/s" font "Times, 26"')
    gp('set ylabel "Time [secs]" font "Times, 26"')
    gp('set xrange [1000:1.e+09]')
    #gp('set yrange [0:]')
    gp('set logscale x 10')
    gp('set logscale y 10')
    gp('set output "| epstopdf --filter > ' + output + '"')

    # legend
    gp('set key width -3.5 samplen 1.8')
    gp('set key top left')

    # margins
    gp('set tmargin .5')
    gp('set rmargin 2.5')
    gp('set lmargin 6.')
    #gp('set lmargin 7.')

    plotdot = Gnuplot.Data(dot, with_='lines lt 4 lw 4.0', title="Parallel DDOT")
    plotsuperacc = Gnuplot.Data(superacc, with_='lines lt 1 lw 4.0', title="Superacc")
    plotfpe3 = Gnuplot.Data(fpe3, with_='lines lt 5 lw 4.0', title="FPE3 + Superacc")
    plotfpe4 = Gnuplot.Data(fpe4, with_='lines lt 2 lw 4.0', title="FPE4 + Superacc")
    plotfpe8 = Gnuplot.Data(fpe8, with_='lines lt 9 lw 4.0', title="FPE8 + Superacc")
    plotfpe4ee = Gnuplot.Data(fpe4ee, with_='lines lt 8 lw 4.0', title="FPE4EE + Superacc")
    plotfpe6ee = Gnuplot.Data(fpe6ee, with_='lines lt 7 lw 4.0', title="FPE6EE + Superacc")
    plotfpe8ee = Gnuplot.Data(fpe8ee, with_='lines lt 3 lw 4.0', title="FPE8EE + Superacc")

    gp.plot(plotdot, plotsuperacc, plotfpe3, plotfpe4, plotfpe8, plotfpe4ee, plotfpe6ee, plotfpe8ee)
    return

plotNbElementsVSGbsAll(sys.argv[1], sys.argv[2], sys.argv[3])
