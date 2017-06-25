#!/bin/sh

/usr/bin/bazel build --config opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/core/user_ops:roi_pooling.so

