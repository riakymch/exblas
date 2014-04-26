#!/bin/bash

# Arguments
if [ $# -ne 0 ]
then
  echo "Usage [no parameters]"
  echo $*
  exit 1
fi

nbElements=67108864
rangeMax=700
step=10
alg=1
exe=./../src/main.out
file=../results/InputRange.VS.GBs.AMD.2014.04.25.dat
echo
touch $file
echo -n "" > $file

#naive DDOT
#for ((range=1; range<=${rangeMax}; range+=${step}))
for range in {1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500, 600, 700}
do
    $exe -n $nbElements -r $range -e 0 -a 0 | tee -a $file
done

#DDOT using superaccumulator
for range in {1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500, 600, 700}
do
    $exe -n $nbElements -r $range -e 0 -a 1 | tee -a $file
done

#DDOT using superaccumulator with various FPEs
for ((nbfpe=2; nbfpe<=8; nbfpe+=1))
do
    for range in {1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500, 600, 700}
    do
	$exe -n $nbElements -r $range -e $nbfpe -a 2 | tee -a $file
    done
done

#DDOT using superaccumulator with FPE of format 8 and the early-exit
for range in {1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500, 600, 700}
do
    $exe -n $nbElements -r $range -e 8 -a 3 | tee -a $file
done

#DDOT using superaccumulator with FPE of format 4 and the early-exit
for range in {1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500, 600, 700}
do
    $exe -n $nbElements -r $range -e 4 -a 4 | tee -a $file
done

#DDOT using superaccumulator with FPE of format 6 and the early-exit
for range in {1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500, 600, 700}
do
    $exe -n $nbElements -r $range -e 6 -a 5 | tee -a $file
done
