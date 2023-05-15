nvcc -DCUSP_PATH=$(realpath .) \
     -I . -I ../KTT/Source \
     -l cuda -l ktt -L ../KTT/Build/x86_64_Release/ \
     --linker-options=-rpath,$(realpath ../KTT/Build/x86_64_Release/) \
     -std=c++17 -g -O3 -lineinfo main.cu

[ "$?" -eq "0" ] || exit 1

./a.out
