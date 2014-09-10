#!/bin/bash

# Arguments
if [ $# -ne 0 ]
then
  echo "Usage [no parameters]"
  echo $*
  exit 1
fi

nMax=2048
step=256
exe=./../src/main.out
file=../results/NbElements.VS.Gflops.K20c.2014.06.29.Round.dat

touch $file
echo -n "" > $file

#DGEMM with various FPEs
for alg in 51 31 34 54
do
    for ((range=2; range<=8; range+=1))
    do
        for ((n=256; n<=${nMax}; n+=${step}))
        do
            $exe -m $n -n $n -k $n -r 1 -e $range -a $alg | tee -a $file
        done
    done
done

#DGEMM: 0-mine; 1-amd; 2-nvidia; with superaccs in private memory
for alg in 50 52 53 30 
do
    for ((n=256; n<=${nMax}; n+=${step}))
    do
        $exe -m $n -n $n -k $n -r 1 -e 0 -a $alg | tee -a $file
    done
done

#DDOT using superaccumulator with FPE of format 8 and the early-exit
#for ((nbElements=1024; nbElements<=${nbMax}; nbElements*=${step}))
#do
#    $exe -n $nbElements -r 1 -e 8 -a 3 | tee -a $file
#done
