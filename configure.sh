#!/bin/bash

wget 'https://github.com/HiPerCoRe/KTT/archive/refs/tags/v2.1.zip'

unzip 'v2.1.zip'

wget 'https://github.com/premake/premake-core/releases/download/v5.0.0-beta2/premake-5.0.0-beta2-linux.tar.gz'

tar -xvf 'premake-5.0.0-beta2-linux.tar.gz'

(
    cd 'KTT-2.1'
    ../premake5 gmake
    cd 'Build'
    make config=release_x86_64
)
