
set terminal postscript eps color enhanced "Times" 26
set grid noxtics noytics
set xlabel "Array size" font "Times, 26"
set ylabel "Gacc/s" font "Times, 26"
set xrange [0:]
set yrange [0:]
set output "| epstopdf --filter > ../imgs/exgemm.gops.amd.15.09.21.short.yes.no.yes.no.pdf"

# legend
set key width -3.5 samplen 1.8
set key top left

# margins
set tmargin .5
set rmargin 2.5
set lmargin 6.5

plot "../results/exgemm.gops.amd.15.09.21.short.yes.no.yes.no.txt" using 1:($5) with lines lt 1 lw 4.0 title "Superacc", \
    "" using 1:($6) with lines lt 8 lw 4.0 title "FPE3 + Superacc", \
    "" using 1:($7) with lines lt 2 lw 4.0 title "FPE4 + Superacc", \
    "" using 1:($8) with lines lt 7 lw 4.0 title "FPE8 + Superacc", \
    "" using 1:($9) with lines lt 4 lw 4.0 title "FPE4EE + Superacc", \
    "" using 1:($10) with lines lt 5 lw 4.0 title "FPE6EE + Superacc", \
    "" using 1:($11) with lines lt 3 lw 4.0 title "FPE8EE + Superacc"

