#!/bin/bash

# Arguments
if [ $# -ne 0 ]
then
  echo "Usage [no parameters]"
  echo $*
  exit 1
fi

nbMax=1000000000
step=2
exe=./../src/main.out
file=../results/NbElements.VS.GBs.AMD.2014.04.25.dat

touch $file
echo -n "" > $file

#naive DDOT
for ((nbElements=1024; nbElements<=${nbMax}; nbElements*=${step}))
do
    $exe -n $nbElements -r 1 -e 0 -a 0 | tee -a $file
done

#DDOT using superaccumulator
for ((nbElements=1024; nbElements<=${nbMax}; nbElements*=${step}))
do
    $exe -n $nbElements -r 1 -e 0 -a 1 | tee -a $file
done

#DDOT using superaccumulator with various FPEs
for ((range=2; range<=8; range+=1))
do
    for ((nbElements=1024; nbElements<=${nbMax}; nbElements*=${step}))
    do
	$exe -n $nbElements -r 1 -e $range -a 2 | tee -a $file
    done
done

#DDOT using superaccumulator with FPE of format 8 and the early-exit
for ((nbElements=1024; nbElements<=${nbMax}; nbElements*=${step}))
do
    $exe -n $nbElements -r 1 -e 8 -a 3 | tee -a $file
done

#DDOT using superaccumulator with FPE of format 4 and the early-exit
for ((nbElements=1024; nbElements<=${nbMax}; nbElements*=${step}))
do
    $exe -n $nbElements -r 1 -e 4 -a 4 | tee -a $file
done

#DDOT using superaccumulator with FPE of format 6 and the early-exit
for ((nbElements=1024; nbElements<=${nbMax}; nbElements*=${step}))
do
    $exe -n $nbElements -r 1 -e 6 -a 5 | tee -a $file
done
