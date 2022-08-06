nvcc -DCUSP_PATH=$(realpath .) \
     -I . -I ../KTT/Source \
     -l cuda -l ktt -L ../KTT/Build/x86_64_Debug/ \
     --linker-options=-rpath,$(realpath ../KTT/Build/x86_64_Debug/) \
     -std=c++17 -g -O3 -DKTT_LINE_INFO -lineinfo main.cu

[ "$?" -eq "0" ] || exit 1

./a.out
