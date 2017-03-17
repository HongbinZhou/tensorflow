
This folder contains the C++ source code and Makefile for wordseg prediction demo.

# Aim

This word segmentation demo is for investigate:

- How to load a pre-trained TensorFlow model in C++?

- How to build and link TensorFlow core to our application?

- How to feed input and get the output for the given graph?

- What's the memory consumption and computation result for given model?

The input wordseg model is trained by *toy_train.py*.

This demo didn't focus on the prediction accuracy/correctness. Right now the
inputs (i.e. embeddings of an input setence) are all fake. Of course in a real
application, the input and prediction outputs should be real and correct.

# Build and Run WordSeg demo

1. Clone TensorFlow repo:

    ``` shell
    git clone https://github.com/tensorflow/tensorflow.git ~/TF/tensorflow
    ```

2. Follow tensorflow/contrib/makefile/README.md to build *libtensorflow-core.a*

    ``` shell
    sudo apt-get install autoconf automake libtool curl make g++ unzip zlib1g-dev \
    git python
    tensorflow/contrib/makefile/build_all_linux.sh
    ```

3. Build wordseg demo

    ``` shell
    cd nncode/tensorflow/cpp
    make
    ```

    Note that if your TensorFlow repo doesn't locate on *~/TF/tensorflow*,
    please update the variable *TFREPO_DIR* in Makefile.

4. Run wordseg demo

    Note that we need first make sure *libprotobuf.a* can be found under
    *LD_LIBRARY_PATH*.

    ``` shell
    cd nncode/tensorflow/cpp
    export LD_LIBRARY_PATH=~/TF/tensorflow/tensorflow/contrib/makefile/gen/protobuf/lib
    make run        # run demo
    make massif     # check memory
    make callgrind  # check function call times
    ```

    To set model files by --graph flag:

    ``` sh
    gen/bin/wordseg --graph="your model path"
    ```
