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
        if line.find("Performance = ") != -1 and line.find(str1) != -1 and line.find(str2) != -1:
            if line.find("NbFPE") != -1:
                line = line.split();
                results.append([line[14], float(line[len(line) - 2])])
            else:
                line = line.split();
                results.append([line[14], float(line[len(line) - 2])])

    f.close()
    return results

def readDataFromFileInputRange(filename, str):
    f = open(filename, 'r')
											    
    results = []
    for line in f:
        if line.find(str) != -1:      
            line = line.split();
            results.append([line[8], line[len(line) - 2]])

    f.close()  
    return results
    
def readDataFromFileInputRangeAll(filename, str1, str2):
    f = open(filename, 'r')

    results = []
    for line in f:
        if line.find("Throughput") != -1 and line.find(str1) != -1 and line.find(str2) != -1:
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
    
def readDataFromFileInputRangeAllGacc(filename, str):
    f = open(filename, 'r')

    results = []
    for line in f:
        if line.find(str) != -1:
            if line.find("NbFPE") != -1:
                line = line.split();
                results.append([line[8], line[len(line) - 2]])
            else:
                line = line.split();
                print line[8]
                print line[14]
                gacc = (float(line[8]) / float(line[14])) * 1e-9
                print gacc
                print line[len(line) - 2]
                exit(0)            
                results.append([line[5], line[len(line) - 2]])    

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
    gp('set key width -2. samplen 1.8')
    gp('set key top left')

    # margins
    gp('set tmargin .5')
    gp('set rmargin 1.')
    gp('set lmargin 5.7')

    plot1 = Gnuplot.Data(results, with_='linespoints lt 1 lw 3.0 lc rgb "black" pt 2', title=None)

    gp.plot(plot1)
    return
    
def plotNbElementsVSGbsAll(input, output): 
    #mine = readDataFromFileAll(input, "Alg = 0", "Alg = 0")
    #amd = readDataFromFileAll(input, "Alg = 1", "Alg = 1")
    #nvidia = readDataFromFileAll(input, "Alg = 2", "Alg = 2")
    #sapr = readDataFromFileAll(input, "Alg = 30", "Alg = 30")
    #fpepr2 = readDataFromFileAll(input, "Alg = 31", "NbFPE = 2")
    #fpepr3 = readDataFromFileAll(input, "Alg = 31", "NbFPE = 3")
    #fpepr4 = readDataFromFileAll(input, "Alg = 31", "NbFPE = 4")
    #fpepr5 = readDataFromFileAll(input, "Alg = 31", "NbFPE = 5")
    #fpepr6 = readDataFromFileAll(input, "Alg = 31", "NbFPE = 6")
    #fpepr7 = readDataFromFileAll(input, "Alg = 31", "NbFPE = 7")
    #fpepr8 = readDataFromFileAll(input, "Alg = 31", "NbFPE = 8")  
    #fpelo2 = readDataFromFileAll(input, "Alg = 6", "NbFPE = 2")
    #fpelo3 = readDataFromFileAll(input, "Alg = 6", "NbFPE = 3")
    #fpelo4 = readDataFromFileAll(input, "Alg = 6", "NbFPE = 4")
    #fpelo5 = readDataFromFileAll(input, "Alg = 6", "NbFPE = 5")
    #fpelo6 = readDataFromFileAll(input, "Alg = 6", "NbFPE = 6")
    #fpelo7 = readDataFromFileAll(input, "Alg = 6", "NbFPE = 7")
    #fpelo8 = readDataFromFileAll(input, "Alg = 6", "NbFPE = 8")          
    sagl = readDataFromFileAll(input, "Alg = 50", "Alg = 50")
    fpegl2 = readDataFromFileAll(input, "Alg = 51", "NbFPE = 2")
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
    gp('set grid noxtics noytics')
    gp('set xlabel "Matrix size (m = n = k)" font "Times, 26"')
    gp('set ylabel "Gacc/s" font "Times, 26"')
    gp('set xrange [0:]')
    gp('set yrange [0:]')
    gp('set output "| epstopdf --filter > ' + output + '"')

    # legend
    gp('set key width 0. samplen 1.8')
    gp('set key top left')

    # margins
    gp('set tmargin .5')
    gp('set rmargin 2.5')
    gp('set lmargin 8.')

    #plotmine = Gnuplot.Data(mine, with_='lines lt 4 lw 4.0', title="Mine DGEMM")
    #plotamd = Gnuplot.Data(amd, with_='lines lt 8 lw 4.0', title="AMD DGEMM")
    #plotnvidia = Gnuplot.Data(nvidia, with_='lines lt 5 lw 4.0', title="AMD DGEMM")
    #plotsa = Gnuplot.Data(sapr, with_='lines lt 1 lw 4.0', title="SuperaccPr")
    #plotfpe2 = Gnuplot.Data(fpepr2, with_='lines lt 8 lw 4.0', title="FPEPr 2")
    #plotfpe3 = Gnuplot.Data(fpepr3, with_='lines lt 5 lw 4.0', title="FPEPr 3")
    #plotfpe4 = Gnuplot.Data(fpepr4, with_='lines lt 2 lw 4.0', title="FPEPr 4")
    #plotfpe5 = Gnuplot.Data(fpepr5, with_='lines lt 6 lw 4.0', title="FPEPr 5")
    #plotfpe6 = Gnuplot.Data(fpepr6, with_='lines lt 7 lw 4.0', title="FPEPr 6")
    #plotfpe7 = Gnuplot.Data(fpepr7, with_='lines lt 0 lw 4.0', title="FPEPr 7")
    #plotfpe8 = Gnuplot.Data(fpepr8, with_='lines lt 4 lw 4.0', title="FPEPr 8")
    #plotfpe2 = Gnuplot.Data(fpelo2, with_='lines lt 8 lw 4.0', title="FPELO 2")
    #plotfpe3 = Gnuplot.Data(fpelo3, with_='lines lt 5 lw 4.0', title="FPELO 3")
    #plotfpe4 = Gnuplot.Data(fpelo4, with_='lines lt 2 lw 4.0', title="FPELO 4")
    #plotfpe5 = Gnuplot.Data(fpelo5, with_='lines lt 6 lw 4.0', title="FPELO 5")
    #plotfpe6 = Gnuplot.Data(fpelo6, with_='lines lt 7 lw 4.0', title="FPELO 6")
    #plotfpe7 = Gnuplot.Data(fpelo7, with_='lines lt 0 lw 4.0', title="FPELO 7")
    #plotfpe8 = Gnuplot.Data(fpelo8, with_='lines lt 4 lw 4.0', title="FPELO 8")            
    plotsa = Gnuplot.Data(sagl, with_='lines lt 3 lw 4.0', title="SuperaccGl")
    plotfpe2 = Gnuplot.Data(fpegl2, with_='lines lt 8 lw 4.0', title="FPEGL 2")
    plotfpe3 = Gnuplot.Data(fpegl3, with_='lines lt 5 lw 4.0', title="FPEGL 3")
    plotfpe4 = Gnuplot.Data(fpegl4, with_='lines lt 2 lw 4.0', title="FPEGL 4")
    plotfpe5 = Gnuplot.Data(fpegl5, with_='lines lt 6 lw 4.0', title="FPEGL 5")
    plotfpe6 = Gnuplot.Data(fpegl6, with_='lines lt 7 lw 4.0', title="FPEGL 6")
    plotfpe7 = Gnuplot.Data(fpegl7, with_='lines lt 0 lw 4.0', title="FPEGL 7")
    plotfpe8 = Gnuplot.Data(fpegl8, with_='lines lt 4 lw 4.0', title="FPEGL 8")
    plotfpe4ex = Gnuplot.Data(fpeexgl4, with_='lines lt 9 lw 4.0', title="FPEEXGL 4")
    plotfpe8ex = Gnuplot.Data(fpeexgl8, with_='lines lt 1 lw 4.0', title="FPEEXGL 8")    
    
    #gp.plot(plotmine, plotamd, plotnvidia, plotsapr, plotsagl)
    #gp.plot(plotsa, plotfpe2, plotfpe3, plotfpe4, plotfpe5, plotfpe6, plotfpe7, plotfpe8)
    gp.plot(plotsa, plotfpe2, plotfpe3, plotfpe4, plotfpe5, plotfpe6, plotfpe7, plotfpe8, plotfpe4ex, plotfpe8ex)
    return

def plotInputRangeVSGbs(input, output): 
    results0 = readDataFromFileInputRange(input, "NbFPE = 2")
    results1 = readDataFromFileInputRange(input, "NbFPE = 3")
    results2 = readDataFromFileInputRange(input, "NbFPE = 4")
    results3 = readDataFromFileInputRange(input, "NbFPE = 5")
    results4 = readDataFromFileInputRange(input, "NbFPE = 6")
    results5 = readDataFromFileInputRange(input, "NbFPE = 7")
    results6 = readDataFromFileInputRange(input, "NbFPE = 8")

    # plot the results
    gp = Gnuplot.Gnuplot(persist=1)
    gp('set terminal postscript eps color enhanced "Times" 26')
    gp('set grid noxtics ytics')
    gp('set xlabel "Input Range" font "Times, 26"')
    gp('set ylabel "Gb/s" font "Times, 26"')
    #gp('set label "Expansion 2" textcolor lt -1 at first 50.2, first 11')
    #gp('set logscale x 2')
    #gp('set format x "2^{%L}"')
    #gp('set xtics nomirror')
    #gp('set ytics nomirror')
    #gp('set xrange xtics')
    gp('set xrange [0:1200]')
    gp('set yrange [0:]')
    gp('set output "| epstopdf --filter > ' + output + '"')    

    # legend
    gp('set key width -2. samplen 1.8')
    gp('set key top right')

    # margins
    gp('set tmargin .5')
    gp('set rmargin 2.')
    gp('set lmargin 7.5')

    plot0 = Gnuplot.Data(results0, with_='lines lt 0 lw 3.0', title="Expansion 2")
    plot1 = Gnuplot.Data(results1, with_='lines lt 1 lw 3.0', title="Expansion 3")
    plot2 = Gnuplot.Data(results2, with_='lines lt 2 lw 3.0', title="Expansion 4")
    plot3 = Gnuplot.Data(results3, with_='lines lt 3 lw 3.0', title="Expansion 5")
    plot4 = Gnuplot.Data(results4, with_='lines lt 4 lw 3.0', title="Expansion 6")
    plot5 = Gnuplot.Data(results5, with_='lines lt 5 lw 3.0', title="Expansion 7")
    plot6 = Gnuplot.Data(results6, with_='lines lt 6 lw 3.0', title="Expansion 8")    
    #plot0 = Gnuplot.Data(results0, with_='linespoints lt 0 lw 3.0 lc rgb "black" pt 2', title="Expansion 2")
    #plot1 = Gnuplot.Data(results1, with_='linespoints lt 1 lw 3.0 lc rgb "black" pt 3', title="Expansion 3")
    #plot2 = Gnuplot.Data(results2, with_='linespoints lt 2 lw 3.0 lc rgb "black" pt 4', title="Expansion 4")
    #plot3 = Gnuplot.Data(results3, with_='linespoints lt 3 lw 3.0 lc rgb "black" pt 5', title="Expansion 5")
    #plot4 = Gnuplot.Data(results4, with_='linespoints lt 4 lw 3.0 lc rgb "black" pt 6', title="Expansion 6")
    #plot5 = Gnuplot.Data(results5, with_='linespoints lt 5 lw 3.0 lc rgb "black" pt 7', title="Expansion 7")
    #plot6 = Gnuplot.Data(results6, with_='linespoints lt 6 lw 3.0 lc rgb "black" pt 8', title="Expansion 8")

    gp.plot(plot0, plot1, plot2, plot3, plot4, plot5, plot6)    
    return
    
def plotInputRangeVSGbsAll(input, output):
    ddot = readDataFromFileInputRangeAll(input, "Alg = 0", "Alg = 0")
    superaccs = readDataFromFileInputRangeAll(input, "Alg = 1", "Alg = 1")
    fpe2 = readDataFromFileInputRangeAll(input, "Alg = 2", "NbFPE = 2")
    fpe3 = readDataFromFileInputRangeAll(input, "Alg = 2", "NbFPE = 3")
    fpe4 = readDataFromFileInputRangeAll(input, "Alg = 2", "NbFPE = 4")
    fpe5 = readDataFromFileInputRangeAll(input, "Alg = 2", "NbFPE = 5")
    fpe6 = readDataFromFileInputRangeAll(input, "Alg = 2", "NbFPE = 6")
    fpe7 = readDataFromFileInputRangeAll(input, "Alg = 2", "NbFPE = 7")
    fpe8 = readDataFromFileInputRangeAll(input, "Alg = 2", "NbFPE = 8")  
    fpe4ee = readDataFromFileInputRangeAll(input, "Alg = 4", "NbFPE = 4")
    fpe6ee = readDataFromFileInputRangeAll(input, "Alg = 5", "NbFPE = 6")      
    fpe8ee = readDataFromFileInputRangeAll(input, "Alg = 3", "NbFPE = 8")
    
#     resultTotal = []
#     for data in ddot:
#         if int(data[0]) > 500:
#             break
#         resultTotal.append([data[0], data[1], superaccs.pop(0)[1], \
#                            fpe2.pop(0)[1], fpe3.pop(0)[1], fpe4.pop(0)[1], \
#                            fpe8.pop(0)[1], fpe4ee.pop(0)[1], fpe6ee.pop(0)[1], \
#                            fpe8ee.pop(0)[1]])
#     f = open(input[0:len(input)-3] + "gnuplot.dat", "w")
#     for data in resultTotal:
#         str = ""
#         for i in data:
#             str += i + "\t"
#         str += ".  \n"
#         f.write(str)            
#     f.close()
#     exit
    
    # plot the results
    gp = Gnuplot.Gnuplot(persist=1)
    gp('set terminal postscript eps color enhanced "Times" 26')
    gp('set grid noxtics noytics')
    gp('set xlabel "Dynamic range (bits)" font "Times, 26"')
    gp('set ylabel "Gacc/s" font "Times, 26"')
    #gp('set format x "1e+%.0f"')    
    #gp('set xrange [1:3.3e150]')
    #gp('set xrange [1:1.0e40]')
    gp('set xrange [0:500]')
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
    
    plotDDOT = Gnuplot.Data(ddot, with_='lines lt 4 lw 4.0', title="Parallel DDOT")
    plotSuperaccs = Gnuplot.Data(superaccs, with_='lines lt 1 lw 4.0', title="Superaccumulator")    
    plotfpe2 = Gnuplot.Data(fpe2, with_='lines lt 8 lw 4.0', title="Expansion 2")
    plotfpe3 = Gnuplot.Data(fpe3, with_='lines lt 5 lw 4.0', title="Expansion 3")
    plotfpe4 = Gnuplot.Data(fpe4, with_='lines lt 2 lw 4.0', title="Expansion 4")
    plotfpe5 = Gnuplot.Data(fpe5, with_='lines lt 6 lw 4.0', title="Expansion 5")
    plotfpe6 = Gnuplot.Data(fpe6, with_='lines lt 7 lw 4.0', title="Expansion 6")
    plotfpe7 = Gnuplot.Data(fpe7, with_='lines lt 0 lw 4.0', title="Expansion 7")
    plotfpe8 = Gnuplot.Data(fpe8, with_='lines lt 7 lw 4.0', title="Expansion 8")
    plotfpe4ee = Gnuplot.Data(fpe4ee, with_='lines lt 0 lw 4.0', title="Expansion 4 early-exit")
    plotfpe6ee = Gnuplot.Data(fpe6ee, with_='lines lt 3 lw 4.0', title="Expansion 6 early-exit")        
    plotfpe8ee = Gnuplot.Data(fpe8ee, with_='lines lt 9 lw 4.0', title="Expansion 8 early-exit")    
    
#     gp.plot(plotDDOT, plotSuperaccs, plotfpe2, plotfpe3, plotfpe4, plotfpe5, plotfpe6, plotfpe7, plotfpe8, plotfpe8ee)
    gp.plot(plotDDOT, plotSuperaccs, plotfpe2, plotfpe3, plotfpe4, plotfpe8, plotfpe4ee, plotfpe6ee, plotfpe8ee)
    return    
    
#plotNbElementsVSGbs(sys.argv[1], sys.argv[2])
plotNbElementsVSGbsAll(sys.argv[1], sys.argv[2])

#plotInputRangeVSGbs(sys.argv[1], sys.argv[2])
#plotInputRangeVSGbsAll(sys.argv[1], sys.argv[2])
