#!/bin/bash

# Arguments
if [ $# -ne 0 ]
then
  echo "Usage [no parameters]"
  echo $*
  exit 1
fi

nMax=4096
step=256
exe=./../src/main.out
file=../results/NbElements.VS.Gflops.AMD.2014.06.06.Round.1.dat

touch $file
echo -n "" > $file

#DGEMM: 0-mine; 1-amd; 2-nvidia; 3-with superaccs in private memory
#for alg in 0 1 2 3 7 
#do
#    for ((n=256; n<=${nMax}; n+=${step}))
#    do
#        $exe -m $n -n $n -k $n -r 1 -e 0 -a $alg | tee -a $file
#    done
#done

#DGEMM with various FPEs: 4-superaccs in private; 6-superaccs in local; 8-superaccs in global
for alg in 4 6 8 
do
    for ((range=2; range<=8; range+=1))
    do
        for ((n=256; n<=${nMax}; n+=${step}))
        do
            $exe -m $n -n $n -k $n -r 1 -e $range -a $alg | tee -a $file
        done
    done
done

#DDOT using superaccumulator with FPE of format 8 and the early-exit
#for ((nbElements=1024; nbElements<=${nbMax}; nbElements*=${step}))
#do
#    $exe -n $nbElements -r 1 -e 8 -a 3 | tee -a $file
#done
