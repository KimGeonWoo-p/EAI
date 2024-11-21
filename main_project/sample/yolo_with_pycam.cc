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
#include <raspicam/raspicam_cv.h>

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

// YOLO 출력 디코딩 함수
std::vector<std::tuple<cv::Rect, float, int>> decode_predictions(
    const float* output, int rows, int cols, float conf_threshold, float iou_threshold) {
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    for (int i = 0; i < rows; ++i) {
        const float* detection = &output[i * cols];
        float confidence = detection[4];  // Confidence score
        if (confidence > conf_threshold) {
            // 바운딩 박스 정보
            float center_x = detection[0];
            float center_y = detection[1];
            float width = detection[2];
            float height = detection[3];

            int left = static_cast<int>(center_x - width / 2);
            int top = static_cast<int>(center_y - height / 2);
            int right = static_cast<int>(center_x + width / 2);
            int bottom = static_cast<int>(center_y + height / 2);

            // 클래스 ID 및 신뢰도
            float max_prob = 0;
            int class_id = -1;
            for (int j = 5; j < cols; ++j) {
                if (detection[j] > max_prob) {
                    max_prob = detection[j];
                    class_id = j - 5;
                }
            }

            // 저장
            boxes.emplace_back(left, top, width, height);
            confidences.push_back(max_prob);
            class_ids.push_back(class_id);
        }
    }

    // NMS (OpenCV 제공)
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, indices);

    // 결과 저장
    std::vector<std::tuple<cv::Rect, float, int>> results;
    for (int idx : indices) {
        results.emplace_back(boxes[idx], confidences[idx], class_ids[idx]);
    }

    return results;
}

void visualize_results(cv::Mat& image, const std::vector<std::tuple<cv::Rect, float, int>>& results) {
  for (const auto& result : results) {
    const cv::Rect& box = std::get<0>(result);       // 바운딩 박스
    const float& confidence = std::get<1>(result);  // 신뢰도
    const int& class_id = std::get<2>(result);      // 클래스 ID

    // 바운딩 박스 그리기
    cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);

    // 텍스트 표시
    std::string label = "Class: " + std::to_string(class_id) + " Conf: " + std::to_string(confidence);
    cv::putText(image, label, cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
  }
}

int main(int argc, char* argv[]) {
  if (argc  != 3) {
    fprintf(stderr, "minimal <tflite model>\n");
    return 1;
  }
  const char* filename = argv[1];
  int CAMSIZE = atoi(argv[2]);

  // (1) Pycam setting
  raspicam::RaspiCam_Cv camera;
  camera.set(cv::CAP_PROP_FORMAT, CV_8UC3);
  camera.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  camera.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  if (!camera.open()) {
    cerr << "Error opening the camera" << endl;
    return 1;
  }
  printf("비디오 세팅 완료! 해상도: %d\n", CAMSIZE);

  while (1) {
    // (2) Load image from Pycam
    cv::Mat image;
    camera.grab();
    camera.retrieve(image);
    printf("카메라에서 이미지 불러옴\n");
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::imshow("Yolo example with Pycam", image);
    if (image.empty()) {
      cerr << "Error capturing image" << endl;
      break;
    }

    try {
      cv::resize(image, image, cv::Size(CAMSIZE, CAMSIZE), 0, 0, cv::INTER_LINEAR);
    } catch (const cv::Exception& e) {
      std::cerr << "OpenCV Exception: " << e.what() << "\n";
      return -1;
    }
    printf("이미지 로드!\n");

    // (3) Load model
    std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
    TFLITE_MINIMAL_CHECK(model != nullptr);
    printf("모델 로드 완료\n");

    // (4) Build interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);
    printf("인터프리터 빌드 완료\n");

    // (5) Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
    printf("=== Pre-invoke Interpreter State ===\n");

    // (6) Push image to input tensor
    auto input_tensor = interpreter->typed_input_tensor<float>(0);

    for (int i=0; i<CAMSIZE; i++){
      for (int j=0; j<CAMSIZE; j++){   
        cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
	for (int k=0; k<3; k++)
          *(input_tensor + i * CAMSIZE*3 + j * 3 + k) = ((float)pixel[k])/255.0;
      }
    }

    printf("인풋텐서 생성 완료!\n");

    // (7) Run inference
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
    printf("\n\n=== Post-invoke Interpreter State ===\n");
    printf("추론 실행!\n");

    // (8) Output parsing
    TfLiteTensor* output_tensor = interpreter->output_tensor(0);

    // 데이터 접근
    const float* output_data = output_tensor->data.f;

    // 행(row) 수와 열(col) 수 계산
    int rows = output_tensor->dims->data[0];  // Total rows (e.g., 1575)
    int cols = output_tensor->dims->data[1];  // Attributes per box (e.g., 85)

    constexpr float conf_threshold = 0.5;
    constexpr float iou_threshold = 0.4;

    // (9) YOLO 출력 디코딩
    auto results = decode_predictions(output_data, rows, cols, conf_threshold, iou_threshold);

    // (10) 결과 시각화
    visualize_results(image, results);
/*
    yolo_output_parsing(cls_tensor, loc_tensor);
    printf("결과 parsing!\n");

    // (9) Output visualize
    yolo_output_visualize(image);
    printf("결과 시각화!\n");
*/
    char key = cv::waitKey(2);
    if (key == 'q') {
        break;
    }
  }

  // (11) release
  camera.release();
	cv::destroyAllWindows();
  return 0;
}

  
