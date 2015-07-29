set terminal postscript eps color enhanced "Times" 26
set grid noxtics noytics
set xlabel "Dynamic range" font "Times, 26"
set ylabel "Gacc/s" font "Times, 26"
#set xrange [1:3.3e150]
set xrange [0:]
#set xtics nomirror rotate by -15
#set logscale x 10
set yrange [0:]
set output "| epstopdf --filter > ../imgs/exsum.range.mic.15.07.28.pdf"

# legend
set key width -7. samplen 1.8
set key center right

# margins
set tmargin .5
set rmargin 3.
set lmargin 6.5

plot "../results/exsum.range.mic.15.07.28.txt" using ($10):(1.053/$8) with lines lt 1 lw 4.0 title "Superacc", \
    "" using ($10):(1.053/$9) with lines lt 8 lw 4.0 title "FPE2 + Superacc", \
    "" using ($10):(1.053/$11) with lines lt 5 lw 4.0 title "FPE3 + Superacc", \
    "" using ($10):(1.053/$13) with lines lt 2 lw 4.0 title "FPE4 + Superacc", \
    "" using ($10):(1.053/$14) with lines lt 3 lw 4.0 title "FPE8EE + Superacc"

