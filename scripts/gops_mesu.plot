set terminal postscript eps color enhanced "Times" 26
set grid noxtics noytics
set xlabel "Array size" font "Times, 26"
set ylabel "Gacc/s" font "Times, 26"
set xrange [1000:1.e+09]
set logscale x 10
set yrange [0:]
set output "| epstopdf --filter > ../imgs/mpi.np1.n24.15.07.28.pdf"

# legend
set key width -3.5 samplen 1.8
set key top right

# margins
set tmargin .5
set rmargin 2.5
set lmargin 6.5

plot "../results/mpi.np1.n24.15.07.28.txt" using 1:(2.6/$3) with lines lt 1 lw 4.0 title "Superacc", \
    "" using 1:(2.6/$4) with lines lt 8 lw 4.0 title "FPE2 + Superacc", \
    "" using 1:(2.6/$5) with lines lt 2 lw 4.0 title "FPE4 + Superacc", \
    "" using 1:(2.6/$8) with lines lt 3 lw 4.0 title "FPE8EE + Superacc"

#plot "../results/.txt" using 1:($20) with lines lt 4 lw 4.0 title "Parallel FP sum", \
#    "" using 1:($22) with lines lt 5 lw 4.0 title "Serial FP sum", \
#    "" using 1:($6) with lines lt 7 lw 4.0 title "TBB deterministic", \
#    "" using 1:($8) with lines lt 0 lw 4.0 title "Demmel fast", \
#    "" using 1:($10) with lines lt 1 lw 4.0 title "Superacc", \
#    "" using 1:($12) with lines lt 8 lw 4.0 title "FPE2 + Superacc", \
#    "" using 1:($16) with lines lt 2 lw 4.0 title "FPE4 + Superacc", \
#    "" using 1:($18) with lines lt 3 lw 4.0 title "FPE8EE + Superacc"

