nvcc -DCUSP_PATH='/home/bigno/school/cusp-autotuned' -I. -I../KTT/Source -lcuda -lktt -L ../KTT/Build/x86_64_Debug/ -std=c++17 main.cu -g 

[ "$?" -eq "0" ] || exit 1

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../KTT/Build/x86_64_Debug/ ./a.out
