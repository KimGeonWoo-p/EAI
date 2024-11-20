/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdlib>

#include <cstdio>
#include <vector>
#include <iostream>
#include <fstream>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "opencv2/opencv.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "yolo_with_pycam.h"
// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: ./minimal_yolo <tflite model>

using namespace std;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {
  if (argc  < 3) {
    fprintf(stderr, "minimal <tflite model>\n");
    return 1;
  }
  const char* filename = argv[1];
  int CAMSIZE = atoi(argv[2]);
  
  // (1) Camera setting
  
  // for video
  cv::VideoCapture video(0);
  if (!video.isOpened())
  {
    cout << "Unable to get video from the camera!" << endl;
    return -1;
  }

  video.set(cv::CAP_PROP_FRAME_WIDTH, CAMSIZE);
  video.set(cv::CAP_PROP_FRAME_HEIGHT, CAMSIZE);
  cv::Mat image;

  // (2) Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // (3) Build interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // (4) Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");

  // (5) load image
  while (video.read(image)) {
    if (image.empty()) {
      cerr << "Error capturing image" << endl;
      break;
    }
    cv::imshow("Yolo example with Pycam", image);

    //vector<cv::Mat> input;
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(CAMSIZE, CAMSIZE));
    //input.push_back(image);

    // (6) Push image to input tensor
    auto input_tensor = interpreter->typed_input_tensor<float>(0);
    for (int i=0; i<CAMSIZE; i++){
      for (int j=0; j<CAMSIZE; j++){   
        cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
        *(input_tensor + i * CAMSIZE*3 + j * 3) = ((float)pixel[0])/255.0;
        *(input_tensor + i * CAMSIZE*3 + j * 3 + 1) = ((float)pixel[1])/255.0;
        *(input_tensor + i * CAMSIZE*3 + j * 3 + 2) = ((float)pixel[2])/255.0;
      }
    }
    // (7) Run inference
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
    printf("\n\n=== Post-invoke Interpreter State ===\n");

    // (8) Output parsing
    TfLiteTensor* cls_tensor = interpreter->output_tensor(1);
    TfLiteTensor* loc_tensor = interpreter->output_tensor(0);
    yolo_output_parsing(cls_tensor, loc_tensor);

    // (9) Output visualize
    yolo_output_visualize(image);

    char key = cv::waitKey(1);
    if (key == 'q') {
        break;
    }
  }
  // (10) release
  cv::destroyAllWindows();
  video.release();
  return 0;
}

  
