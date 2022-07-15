nvcc -DCUSP_PATH='/home/bigno/school/cusp-autotuned' -I. -I../KTT/Source -lcuda -lktt -O3 \
     -L ../KTT/Build/x86_64_Debug/ --linker-options=-rpath,$(realpath ../KTT/Build/x86_64_Debug/) \
     -std=c++17 -g -lineinfo main.cu

[ "$?" -eq "0" ] || exit 1

./a.out
