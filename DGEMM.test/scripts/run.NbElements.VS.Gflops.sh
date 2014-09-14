#!/bin/bash

# Arguments
if [ $# -ne 0 ]
then
  echo "Usage [no parameters]"
  echo $*
  exit 1
fi

begin=256
step=256
nMax=2048
exe=./../src/main.out.amd
file=../results/NbElements.VS.Gflops.Fusion.2014.09.14.Round.dat

touch $file
echo -n "" > $file

for alg in 0 2 
do
    for ((n=${begin}; n<=${nMax}; n+=${step}))
    do
        $exe -m $n -n $n -k $n -r 1 -e 0 -a $alg | tee -a $file
    done
done

#for alg in 54 34
#do
#    for ((range=3; range<=8; range+=i))
#    do
#     	for ((ml=2; ml<=8; range+=1))
#	do
#	    for ((n=${begin}; n<=${nMax}; n+=${step}))
#            do
#                $exe -m $n -n $n -k $n -r 1 -e $range -a $alg -ml $ml | tee -a $file
#            done
#	done
#    done
#done

for alg in 30 32 33 50 52 53
do
    for ((n=${begin}; n<=${nMax}; n+=${step}))
    do
        $exe -m $n -n $n -k $n -r 1 -e 0 -a $alg | tee -a $file
    done
done

#DGEMM with various FPEs
for alg in 51 31
do
    for ((range=3; range<=8; range+=1))
    do
        for ((n=${begin}; n<=${nMax}; n+=${step}))
        do
            $exe -m $n -n $n -k $n -r 1 -e $range -a $alg | tee -a $file
        done
    done
done
