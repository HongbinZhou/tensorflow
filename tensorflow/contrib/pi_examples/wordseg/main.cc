#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"


#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

using namespace std;
using namespace chrono;
using namespace tensorflow;

int main(int argc, char* argv[]) {
  cout << "hello wordseg!" << endl;

  // command-line flags
  string graph = "/home/hbzhou/E/repo/nncode/tensorflow/model_embedding_blstm/toyws.pb";
  std::vector<tensorflow::Flag> flag_list = {
      Flag("graph", &graph, "graph to be executed"),
  };
  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << "\n" << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(usage.c_str(), &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  // Initialize a tensorflow session
  cout << "start initalize session" << "\n";
  Session* session;
  Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    cout << status.ToString() << "\n";
    return 1;
  }

  // Read in the protobuf graph we exported
  // (The path seems to be relative to the cwd. Keep this in mind
  // when using `bazel run` since the cwd isn't where you call
  // `bazel run` but from inside a temp folder.)
  GraphDef graph_def;

  // for batch normalization model
  // status = ReadBinaryProto(Env::Default(), "/home/hbzhou/E/repo/nncode/tensorflow/model_20170309/toyws_opt.pb", &graph_def);

  // for non batch norm
  cout << "load graph: " << graph << endl;
  status = ReadBinaryProto(Env::Default(), graph, &graph_def);

  // status = ReadBinaryProto(Env::Default(), "tensorflow/contrib/pi_examples/wordseg/models/frozen_graph.pb", &graph_def);
  if (!status.ok()) {
    cout << status.ToString() << "\n";
    return 1;
  }

  // Add the graph to the session
  status = session->Create(graph_def);
  if (!status.ok()) {
    cout << status.ToString() << "\n";
    return 1;
  }

  cout << "preparing input data..." << endl;

  // for (int i=0; i < msg.size(); ++i)
  //   std::cout << msg[i] << ' ';

  int max_setence_len = 100;
  int batch_size = 1;
  int nTests = batch_size;
  int embedding_size = 200;
  Tensor x(DT_FLOAT, TensorShape({nTests, max_setence_len, embedding_size}));
  auto dst = x.tensor<float, 3>();
  for (int i = 0; i < nTests; i++) {
    for (int j = 0; j < max_setence_len; j++){
      for (int k = 0; k < embedding_size; k++){
        dst(i, j, k) = 0.1f;
      }
    }
  }
  Tensor seqlen(DT_INT32, TensorShape({nTests}));
  auto dst2 = seqlen.flat<int>().data();
  dst2[0] = max_setence_len;
  // auto dst = seqlen.tensor<int>();
  // for (int i = 0; i < nTests; i++) {
  //     dst(i) = max_setence_len;
  //   }
  // }


  cout << "data is ready" << endl;
  vector<pair<string, Tensor>> inputs = {
    { "input/data/x", x},
    { "input/data/seqlen", seqlen}
  };

  // The session will initialize the outputs
  vector<Tensor> outputs;

  // Run the session, evaluating our "softmax" operation from the graph
  status = session->Run(inputs, {"toyws_output/toyws_output"}, {}, &outputs);
  if (!status.ok()) {
    cout << status.ToString() << "\n";
    return 1;
  }else{
  	cout << "Success load graph !! " << "\n";
  }


  for (vector<Tensor>::iterator it = outputs.begin() ; it != outputs.end(); ++it) {
    // auto items = it->shaped<int, 2>({nTests, max_setence_len});
    // TODO: can't use output???
  }

  return 0;
}
