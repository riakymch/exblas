set terminal postscript eps color enhanced "Times" 26
set grid noxtics noytics
set xlabel "Dynamic range" font "Times, 26"
set ylabel "Gacc/s" font "Times, 26"
set xrange [1:3.3e150]
set xtics nomirror rotate by -15
set logscale x 10
set yrange [0:]
set output "| epstopdf --filter > ../imgs/InputRange.VS.GBs.AMD.2014.04.29.Round.2.pdf"

# legend
set key width -7. samplen 1.8
set key top right

# margins
set tmargin .5
set rmargin 3.
set lmargin 6.5

plot "../results/InputRange.VS.GBs.AMD.2014.04.29.Round.gnuplot.dat" using (2**$1):($2) with lines lt 4 lw 4.0 title "Parallel DDOT", \
    "" using (2**$1):($3) with lines lt 1 lw 4.0 title "Superaccumulator", \
    "" using (2**$1):($4) with lines lt 8 lw 4.0 title "Expansion 2", \
    "" using (2**$1):($5) with lines lt 5 lw 4.0 title "Expansion 3", \
    "" using (2**$1):($6) with lines lt 2 lw 4.0 title "Expansion 4", \
    "" using (2**$1):($7) with lines lt 7 lw 4.0 title "Expansion 8", \
    "" using (2**$1):($8) with lines lt 0 lw 4.0 title "Expansion 4 early-exit", \
    "" using (2**$1):($9) with lines lt 3 lw 4.0 title "Expansion 6 early-exit", \
    "" using (2**$1):($10) with lines lt 9 lw 4.0 title "Expansion 8 early-exit"
