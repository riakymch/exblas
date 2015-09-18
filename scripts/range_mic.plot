set terminal postscript eps color enhanced "Times" 26
set grid noxtics noytics
set xlabel "Dynamic range" font "Times, 26"
set ylabel "Gacc/s" font "Times, 26"
set xrange [1:3.3e150]
set xtics nomirror rotate by -15
set logscale x 10
set yrange [0:25]
set output "| epstopdf --filter > ../imgs/exsum.range.mic.15.09.14.pdf"

# legend
set key width -3.5 samplen 1.8
set key center right

# margins
set tmargin .5
set rmargin 3.
set lmargin 6.5

plot "../results/exsum.range.mic.15.09.14.txt" using (2**$2):(1.053/$3) with lines lt 1 lw 4.0 title "Superacc", \
    "" using (2**$2):(1.053/$4) with lines lt 8 lw 4.0 title "FPE2 + Superacc", \
    "" using (2**$2):(1.053/$5) with lines lt 2 lw 4.0 title "FPE4 + Superacc", \
    "" using (2**$2):(1.053/$8) with lines lt 3 lw 4.0 title "FPE8EE + Superacc"

