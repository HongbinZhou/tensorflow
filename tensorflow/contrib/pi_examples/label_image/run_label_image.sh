TF_ROOT=.
export LD_LIBRARY_PATH=${TF_ROOT}/tensorflow/contrib/makefile/gen/protobuf/lib

BIN=${TF_ROOT}/tensorflow/contrib/pi_examples/label_image/gen/bin/label_image

echo "run $BIN ..."
$BIN
echo "run $BIN done!"

