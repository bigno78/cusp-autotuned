nvcc -I. -I../KTT/Source -lcuda -lktt -L ../KTT/Build/x86_64_Release/ -std=c++17 main.cu -g 

[ "$?" -eq "0" ] || exit 1

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../KTT/Build/x86_64_Release/ ./a.out
