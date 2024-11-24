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

using namespace std;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

using namespace std;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

float calculate_iou(const BoundingBox& box1, const BoundingBox& box2) {
    // 두 박스의 교집합 좌표 계산
    float x1 = std::max(box1.x - box1.width / 2, box2.x - box2.width / 2);
    float y1 = std::max(box1.y - box1.height / 2, box2.y - box2.height / 2);
    float x2 = std::min(box1.x + box1.width / 2, box2.x + box2.width / 2);
    float y2 = std::min(box1.y + box1.height / 2, box2.y + box2.height / 2);

    // 교집합 영역 계산
    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);

    // 합집합 영역 계산
    float union_area = box1.width * box1.height + box2.width * box2.height - intersection;

    // IoU 반환
    return intersection / union_area;
}

// NMS 수행 함수
// NMS 함수
std::vector<int> non_max_suppression(const std::vector<BoundingBox>& boxes, float iou_threshold) {
    std::vector<int> indices;
    std::vector<bool> suppressed(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) continue;

        indices.push_back(i);
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed[j]) continue;
            
            // IoU 계산
            float iou = calculate_iou(boxes[i], boxes[j]);
            if (iou > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }
    return indices;
}

// YOLO 디코딩 함수
std::vector<BoundingBox> decode_predictions(const float* output, int num_detections, int num_classes, float conf_threshold, float iou_threshold, int image_width, int image_height) {
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    for (int i = 0; i < num_detections; ++i) {
        const float* detection = output + i * (num_classes + 5);

        float confidence = detection[4];
        if (confidence > conf_threshold) {
            float x = detection[0];       // 중심 x 좌표 (정규화)
            float y = detection[1];       // 중심 y 좌표 (정규화)
            float width = detection[2];   // 박스 너비 (정규화)
            float height = detection[3];  // 박스 높이 (정규화)
/*
            // 디버깅용 로그
            std::cout << "Raw Bounding Box: x=" << x
                      << ", y=" << y
                      << ", width=" << width
                      << ", height=" << height << std::endl;
*/
            // 클래스 ID 및 클래스 점수
            float max_class_score = 0.0f;
            int class_id = -1;
            for (int j = 0; j < num_classes; ++j) {
                float class_score = detection[5 + j];
                if (class_score > max_class_score) {
                    max_class_score = class_score;
                    class_id = j;
                }
            }
            
            float final_score = confidence * max_class_score;
            if (final_score > conf_threshold) {
                // 좌표 변환: 정규화된 값을 이미지 크기 기준으로 변환
                int x_pos = static_cast<int>(x * image_width);
                int y_pos = static_cast<int>(y * image_height);
                int box_width = static_cast<int>(width * image_width);
                int box_height = static_cast<int>(height * image_height);


                // OpenCV 박스 저장
                boxes.emplace_back(x_pos, y_pos, box_width, box_height);
                confidences.push_back(final_score);
                class_ids.push_back(class_id);
            }
        }
    }

    // NMS 적용
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, indices);

    std::vector<BoundingBox> results;
    for (int idx : indices) {
        const auto& box = boxes[idx];
        BoundingBox bbox;
        bbox.x = box.x;
        bbox.y = box.y;
        bbox.width = box.width;
        bbox.height = box.height;
        bbox.confidence = confidences[idx];
        bbox.class_id = class_ids[idx];
        results.push_back(bbox);
    }

    return results;
}

// 바운딩 박스를 이미지에 그리는 함수
cv::Mat draw_detections(cv::Mat img, const std::vector<BoundingBox>& detections, int image_width, int image_height) {
    for (const auto& box : detections) {
       // 디버깅: 원본 좌표와 크기 출력
       std::cout << "Raw Detection: x=" << box.x
         << ", y=" << box.y
         << ", width=" << box.width
         << ", height=" << box.height
         << ", confidence=" << box.confidence
         << ", class_id=" << box.class_id << std::endl;

        // 바운딩 박스 좌표 계산
        int x1 = static_cast<int>((box.x - box.width / 2));
        int y1 = static_cast<int>((box.y - box.height / 2));
        int x2 = static_cast<int>((box.x + box.width / 2));
        int y2 = static_cast<int>((box.y + box.height / 2));

        // 좌표 클리핑
        x1 = std::max(0, std::min(image_width - 1, x1));
        y1 = std::max(0, std::min(image_height - 1, y1));
        x2 = std::max(0, std::min(image_width - 1, x2));
        y2 = std::max(0, std::min(image_height - 1, y2));

        // 디버깅: 계산된 좌표 출력
        std::cout << "Calculated Coordinates: x1=" << x1
                  << ", y1=" << y1
                  << ", x2=" << x2
                  << ", y2=" << y2 << std::endl;

        // 바운딩 박스 그리기
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
        if (x2 <= x1 || y2 <= y1) {
            std::cerr << "Invalid box size: width=" << (x2 - x1)
                      << ", height=" << (y2 - y1) << std::endl;
            return img;

        }

        // 라벨 그리기
        std::ostringstream label;
        label << "Class " << box.class_id << ": " << std::fixed << std::setprecision(2) << box.confidence;
        cv::putText(img, label.str(), cv::Point(x1, y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }

    return img;
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
  camera.set(cv::CAP_PROP_FRAME_WIDTH, CAMSIZE);
  camera.set(cv::CAP_PROP_FRAME_HEIGHT, CAMSIZE);
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
    TfLiteTensor* output_tensor = interpreter->tensor(interpreter->outputs()[0]);
    const float* output_data = output_tensor->data.f; // 출력 데이터 접근

    // 출력 텐서 크기 정보
    int num_detections = output_tensor->dims->data[1];
    int num_classes = output_tensor->dims->data[2] - 5;

    // YOLO 디코딩
    float conf_threshold = 0.5;
    float iou_threshold = 0.4;

    std::vector<BoundingBox> results = decode_predictions(output_data, num_detections, num_classes, conf_threshold, iou_threshold, CAMSIZE, CAMSIZE);

    // (9) Output visualize
    image = draw_detections(image, results, CAMSIZE, CAMSIZE);

    cv::imshow("Yolo example with Pycam", image);

    char key = cv::waitKey(1);
    if (key == 'q') {
        break;
    }
  }

  // (11) release
  camera.release();
        cv::destroyAllWindows();
  return 0;
}

