#!/bin/bash

# Arguments
if [ $# -ne 0 ]
then
  echo "Usage [no parameters]"
  echo $*
  exit 1
fi

begin=2048
step=512
nMin=512
exe=./../src/main.out.amd
file=../results/NbElements.VS.Gflops.Tahiti.2014.09.19.Round.dat

touch $file
echo -n "" > $file

#for alg in 30 32 33 50 52 53
#do
#    for ((n=${begin}; n<=${nMin}; n+=${step}))
#    do
#        $exe -m $n -n $n -k $n -r 1 -e 0 -a $alg | tee -a $file
#    done
#done

#DGEMM with various FPEs
for alg in 31
do
    for ((range=3; range<=5; range+=1))
    do
      for ((multi=16; range>=1; range/=2))
      do
        for ((n=${begin}; n=>${nMin}; n-=${step}))
        do
            $exe -m $n -n $n -k $n -r 1 -e $range -a $alg -ml $multi | tee -a $file
        done
      done
    done
done

for alg in 32 34
do
    for ((multi=16; range>=1; range/=2))
    do
	for ((n=${begin}; n=>${nMin}; n-=${step}))
        do
            $exe -m $n -n $n -k $n -r 1 -e 3 -a $alg -ml $multi | tee -a $file
        done
    done
done

for alg in 0 2 
do
    for ((n=${begin}; n=>${nMin}; n-=${step}))
    do
        $exe -m $n -n $n -k $n -r 1 -e 0 -a $alg | tee -a $file
    done
done

