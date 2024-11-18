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
#include <cstdio>
#include <vector>
#include <iostream>
#include <fstream>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// for camura
#include <opencv2/opencv.hpp>
#include <unistd.h> // sleep 함수

// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: minimal <tflite model>

using namespace std;

#define MNIST_INPUT "../mnist_dataset/mnist_images"
#define MNIST_LABEL "../mnist_dataset//mnist_labels"

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist(const cv::Mat& mat, vector<vector<float>>& input_vec) {
	for (int r = 0; r < 28; ++r){
        	input_vec.push_back(vector<float>());
		for (int c = 0; c < 28; ++c)
			input_vec[r].push_back((float)mat.at<uchar>(r, c));
	}
}

int main(int argc, char* argv[]) {
  const char* filename = "mnist.tflite";
  // Load model
  unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // build interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

  printf("=== Pre-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // for video
  cv::VideoCapture video(0);
  if (!video.isOpened())
  {
    cout << "Unable to get video from the camera!" << endl;
    return -1;
  }

  video.set(cv::CAP_PROP_FRAME_WIDTH, 28);
  cv::Mat orig;
  cv::Mat gray;

  while (video.read(orig))
  {
    cv::cvtColor(orig, gray, cv::COLOR_BGR2GRAY);

    auto input_tensor = interpreter->typed_input_tensor<float>(0);

    vector<vector<float>> input_vector;
    read_Mnist(gray, input_vector);

    for(int i=0; i<28; ++i) // image rows
      for(int j=0; j<28; ++j) // image cols
        input_tensor[i * 28 + j] = input_vector[i][j] / 255.0; // normalize and copy input values.

    // Run inference
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

    auto output_tensor = interpreter->typed_output_tensor<float>(0);

    float max = -1;
    int max_i = 0;
    for(int i=0; i<10; ++i) {
      printf("label:%d, pred:%f\n", i, output_tensor[i]);
    }

    // 결과 출력
    printf("\n\n\n\n");
    // 벡터를 OpenCV Mat 객체로 변환
    cv::Mat image(28, 28, CV_8UC1); // 8-bit, 1채널 (흑백 이미지)
    for (int i = 0; i < 28; ++i) {
      for (int j = 0; j < 28; ++j) {
          image.at<uchar>(i, j) = static_cast<uchar>(input_vector[i][j]);
      }
    }

    // 이미지 시각화
    cv::imshow("Pixel Intensity", image);
    
    // 1초 정지
    sleep(0.1);

    if (cv::waitKey(25) >= 0)
      break;
  }

  cv::destroyAllWindows();
  video.release();

  return 0;
}
