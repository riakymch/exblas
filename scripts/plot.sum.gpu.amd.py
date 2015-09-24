#!/usr/bin/env python
import Gnuplot
import math
import sys

def readDataFromFileInputRangeAll(filename, str1, str2):
    f = open(filename, 'r')

    results = []
    for line in f:
        if line.find(str1) != -1 and line.find(str2) != -1:
            if line.find("NbFPE") != -1:
                line = line.split();
                #if float(line[8]) > 500:
                #    break
                #results.append([line[8], line[len(line) - 2]])
                results.append([line[8], str(float(line[len(line) - 2]) / 8.)])
            else:
                line = line.split();
                #if float(line[5]) > 500:
                #    break
                #results.append([line[5], line[len(line) - 2]])
                results.append([line[5], str(float(line[len(line) - 2]) / 8.)])

    f.close()
    return results

def plotNbElementsVSGbsAll(input1, input2, output):
    f = open(input2, 'r')

    red = []
    demmel = []
    superacc = []
    fpe2 = []
    fpe3 = []
    fpe4 = []
    fpe8 = []
    fpe8ee = []
    f2 = open(input1, 'r')
    for line in f2:
        if line.find("Alg = 2") != -1:
            line = line.split()
            red.append([line[8], float(line[len(line) - 2]) / 8.])
    print red
    f2.close()
    for line in f:
        '''if line.find("Alg = 1") != -1:
            line = line.split()
            superacc.append([line[11], float(line[len(line) - 2]) / 8.])
        elif line.find("Alg = 2") != -1 and line.find("NbFPE = 2") != -1:
            line = line.split()
            fpe2.append([line[11], float(line[len(line) - 2]) / 8.])
        elif line.find("Alg = 2") != -1 and line.find("NbFPE = 3") != -1:
            line = line.split()
            fpe3.append([line[11], float(line[len(line) - 2]) / 8.])
        elif line.find("Alg = 2") != -1 and line.find("NbFPE = 4") != -1:
            line = line.split()
            fpe4.append([line[11], float(line[len(line) - 2]) / 8.])
        elif line.find("Alg = 2") != -1 and line.find("NbFPE = 8") != -1:
            line = line.split()
            fpe8.append([line[11], float(line[len(line) - 2]) / 8.])
        elif line.find("Alg = 5") != -1:
            line = line.split()
            fpe8ee.append([line[11], float(line[len(line) - 2]) / 8.])
        elif line.find("Alg = 6") != -1:
            line = line.split()
            demmel.append([line[11], float(line[len(line) - 2]) / 8.])'''
        line = line.split()
        mult = 1e-9*float(line[0])
        superacc.append([line[0], mult / float(line[2])]);
        fpe2.append([line[0], mult / float(line[3])]);
        fpe3.append([line[0], mult / float(line[4])]);
        fpe4.append([line[0], mult / float(line[5])]);
        fpe8.append([line[0], mult / float(line[6])]);
        fpe8ee.append([line[0], mult / float(line[7])]);
    f.close()

    # plot the results
    gp = Gnuplot.Gnuplot(persist=1)
    gp('set terminal postscript eps color enhanced "Times" 26')
    gp('set grid noxtics noytics')
    gp('set xlabel "Array size" font "Times, 26"')
    gp('set ylabel "Gacc/s" font "Times, 26"')
    gp('set xrange [1000:1.e+09]')
    gp('set logscale x 10')
    gp('set yrange [0:]')
    gp('set output "| epstopdf --filter > ' + output + '"')

    # legend
    gp('set key width -3.5 samplen 1.8')
    gp('set key top left')

    # margins
    gp('set tmargin .5')
    gp('set rmargin 2.5')
    gp('set lmargin 6.5')

    plotsuperacc = Gnuplot.Data(superacc, with_='lines lt 1 lw 4.0', title="Superacc")
    plotfpe2 = Gnuplot.Data(fpe2, with_='lines lt 8 lw 4.0', title="FPE2 + Superacc")
    plotfpe3 = Gnuplot.Data(fpe3, with_='lines lt 5 lw 4.0', title="FPE3 + Superacc")
    plotfpe4 = Gnuplot.Data(fpe4, with_='lines lt 2 lw 4.0', title="FPE4 + Superacc")
    plotfpe8 = Gnuplot.Data(fpe8, with_='lines lt 9 lw 4.0', title="FPE8 + Superacc")
    plotfpe8ee = Gnuplot.Data(fpe8ee, with_='lines lt 3 lw 4.0', title="FPE8EE + Superacc")
    plotred = Gnuplot.Data(red, with_='lines lt 4 lw 4.0', title="Parallel FP Sum")

    gp.plot(plotred, plotsuperacc, plotfpe2, plotfpe3, plotfpe4, plotfpe8, plotfpe8ee)
    return

def plotInputRangeVSGbsAll(inputfile, output):
    i = 1;
    fr = open(inputfile, 'r')
    fw = open(inputfile[0:len(inputfile)-3] + "gnuplot.dat", "w")
    strl = ""
    for line in fr:
        if line.find("Alg = 0") != -1:
            line = line.split()
            strl += line[8] + "\t" + str(float(line[len(line) - 2]) / 8.) + "\t"
            i+=1
        elif line.find("Alg = 1") != -1:
            line = line.split()
            strl += str(float(line[len(line) - 2]) / 8.) + "\t"
            i+=1
        elif line.find("Alg = 2") != -1 and line.find("NbFPE = 2") != -1:
            line = line.split()
            strl += str(float(line[len(line) - 2]) / 8.) + "\t"
            i+=1
        elif line.find("Alg = 2") != -1 and line.find("NbFPE = 3") != -1:
            line = line.split()
            strl += str(float(line[len(line) - 2]) / 8.) + "\t"
            i+=1
        elif line.find("Alg = 2") != -1 and line.find("NbFPE = 4") != -1:
            line = line.split()
            strl += str(float(line[len(line) - 2]) / 8.) + "\t"
            i+=1
        elif line.find("Alg = 2") != -1 and line.find("NbFPE = 8") != -1:
            line = line.split()
            strl += str(float(line[len(line) - 2]) / 8.) + "\t"
            i+=1
        elif line.find("Alg = 3") != -1:
            line = line.split()
            strl += str(float(line[len(line) - 2]) / 8.) + "\t"
            i+=1
        elif line.find("Alg = 5") != -1:
            line = line.split()
            strl += str(float(line[len(line) - 2]) / 8.) + "\t"
            i+=1
        elif line.find("Alg = 6") != -1:
            line = line.split()
            strl += str(float(line[len(line) - 2]) / 8.) + "\n"
            i+=1
        if (i % 10 == 0):
            fw.write(strl)
            strl = ""
            i = 1
    fr.close()
    fw.close()

    '''# plot the results
    gp = Gnuplot.Gnuplot(persist=1)
    gp('set terminal postscript eps color enhanced "Times" 26')
    gp('set grid noxtics noytics')
    gp('set xlabel "Dynamic range (bits)" font "Times, 26"')
    gp('set ylabel "Gacc/s" font "Times, 26"')
    gp('set logscale x')
    #gp('set format x "1e+%.0f"')
    #gp('set xrange [1:3.3e150]')
    #gp('set xrange [1:1.0e40]')
    gp('set yrange [0:]')
    gp('set xtics nomirror rotate by -15')
    gp('set output "| epstopdf --filter > ' + output + '"')


    # legend#gp('set logscale x 2')
    gp('set key width -2. samplen 1.8')
    gp('set key top right')    

    # margins
    gp('set tmargin .5')
    gp('set rmargin 2.')
    gp('set lmargin 6.5')

   # plot0 = Gnuplot.Data(results0, with_='lines lt 1 lw 4.0', title="Superaccumulator")
   # plot1 = Gnuplot.Data(results1, with_='lines lt 8 lw 4.0', title="Expansion 2")
   # plot2 = Gnuplot.Data(results5, with_='lines lt 5 lw 4.0', title="Expansion 3")
   # plot3 = Gnuplot.Data(results2, with_='lines lt 2 lw 4.0', title="Expansion 4")
    #plot4 = Gnuplot.Data(results6, with_='lines lt 6 lw 4.0', title="Expansion 5")
    #plot5 = Gnuplot.Data(results7, with_='lines lt 7 lw 4.0', title="Expansion 6")
    #plot6 = Gnuplot.Data(results8, with_='lines lt 0 lw 4.0', title="Expansion 7")
   # plot7 = Gnuplot.Data(results3, with_='lines lt 7 lw 4.0', title="Expansion 8")    
   # plot8 = Gnuplot.Data(results4, with_='lines lt 4 lw 4.0', title="Parallel FP Sum")
   # plot9 = Gnuplot.Data(results9, with_='lines lt 3 lw 4.0', title="Expansion 8 early-exit")

    #gp.plot(plot8, plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot0)
    #gp.plot(plot8, plot1, plot2, plot3, plot7, plot9, plot0)'''
    return

plotNbElementsVSGbsAll(sys.argv[1], sys.argv[2], sys.argv[3])
#plotInputRangeVSGbsAll(sys.argv[1], sys.argv[1])
