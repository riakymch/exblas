
set terminal postscript eps color enhanced "Times" 26
set grid noxtics noytics
set xlabel "Array size" font "Times, 26"
set ylabel "Gacc/s" font "Times, 26"
set xrange [1000:1.e+09]
set logscale x 10
set yrange [0:25]
set output "| epstopdf --filter > ../imgs/exsum.gops.mic.15.09.11.0.pdf"

# legend
set key width -3.5 samplen 1.8
set key top left

# margins
set tmargin .5
set rmargin 2.5
set lmargin 6.5

plot "../results/exsum.gops.mic.15.09.11.0.txt" using 1:(1.053/$3) with lines lt 1 lw 4.0 title "Superacc", \
    "" using 1:(1.053/$4) with lines lt 8 lw 4.0 title "FPE2 + Superacc", \
    "" using 1:(1.053/$5) with lines lt 2 lw 4.0 title "FPE4 + Superacc", \
    "" using 1:(1.053/$8) with lines lt 3 lw 4.0 title "FPE8EE + Superacc"

