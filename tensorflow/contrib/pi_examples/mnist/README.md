

# Requirement
install python3 tensorflow wheel file

# Gen binary model

```sh
cd tensorflow/tensorflow/contrib/pi_examples/mnist
python3 ./mnist.py
# now the models (in pbtxt format) has been writen to folder models
python3 ./freeze_model.py # freeze *.pbtxt -> *.pb
# now we can find the binary model file 'frozen_model.pb' in the models
```

Note that to freeze model, we can use script freeze_graph.py directly. The
output_node_names should be set in the training script, and can also be found in
the *.pbtxt model file.

``` shell
python3 /usr/lib/python3.5/site-packages/tensorflow/python/tools/freeze_graph.py  --input_graph models/graph.pb  --input_checkpoint ./models/model.ckpt  --output_graph ./models/frozen_graph.pb --output_node_names "softmax"
```

# Build C++ executable to consume the binary model

```sh
pwd
# tensorflow/tensorflow/contrib/pi_examples/mnist
cd MNIST_data
for i in $.gz; do gnuzip -d $i; done
# change to tensorflow root directory
pwd
# tensorflow
make -f tensorflow/contrib/pi_examples/mnist/Makefile run
```
# Thanks
[tensorgraph](https://github.com/JackyTung/tensorgraph)
