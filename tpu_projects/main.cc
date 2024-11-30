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

// tpu활용을 위한 것
#include <memory>
#include <chrono>
#include <cassert>
#include "headers/edgetpu_c.h"
#include "opencv2/opencv.hpp"

// gpio
#include "headers/gpio.h"

using namespace std;
using namespace cv;

struct _box{
    int class_id;
    float confidence;
    float x;
    float y;
    float width;
    float height;
} typedef BoundingBox;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

float calculate_iou(const BoundingBox& box1, const BoundingBox& box2) {
    // 두 박스의 교집합 좌표 계산
    float x1 = max(box1.x - box1.width / 2, box2.x - box2.width / 2);
    float y1 = max(box1.y - box1.height / 2, box2.y - box2.height / 2);
    float x2 = min(box1.x + box1.width / 2, box2.x + box2.width / 2);
    float y2 = min(box1.y + box1.height / 2, box2.y + box2.height / 2);

    // 교집합 영역 계산
    float intersection = max(0.0f, x2 - x1) * max(0.0f, y2 - y1);

    // 합집합 영역 계산
    float union_area = box1.width * box1.height + box2.width * box2.height - intersection;

    // IoU 반환
    return intersection / union_area;
}

// NMS 수행 함수
// NMS 함수
vector<int> non_max_suppression(const vector<BoundingBox>& boxes, float iou_threshold) {
    vector<int> indices;
    vector<bool> suppressed(boxes.size(), false);

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
vector<BoundingBox> decode_predictions(const float* output, int num_detections, int num_classes, float conf_threshold, float iou_threshold, int image_width, int image_height) {
    vector<Rect> boxes;
    vector<float> confidences;
    vector<int> class_ids;

    for (int i = 0; i < num_detections; ++i) {
        const float* detection = output + i * (num_classes + 5);

        float confidence = detection[4];
        if (confidence > conf_threshold) {
            float x = detection[0];       // 중심 x 좌표 (정규화)
            float y = detection[1];       // 중심 y 좌표 (정규화)
            float width = detection[2];   // 박스 너비 (정규화)
            float height = detection[3];  // 박스 높이 (정규화)

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

			//만약 threshold보다 확률이 높으면서 포유류인 경우
            if (final_score > conf_threshold && class_id > 1) {
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
    vector<int> indices;
    dnn::NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, indices);

    vector<BoundingBox> results;
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
Mat draw_detections(Mat img, const vector<BoundingBox>& detections, int image_width, int image_height) {
    for (const auto& box : detections) {
       // 디버깅: 원본 좌표와 크기 출력
       cout << "Raw Detection: x=" << box.x
         << ", y=" << box.y
         << ", width=" << box.width
         << ", height=" << box.height
         << ", confidence=" << box.confidence
         << ", class_id=" << box.class_id << endl;

        // 바운딩 박스 좌표 계산
        int x1 = static_cast<int>((box.x - box.width / 2));
        int y1 = static_cast<int>((box.y - box.height / 2));
        int x2 = static_cast<int>((box.x + box.width / 2));
        int y2 = static_cast<int>((box.y + box.height / 2));

        // 좌표 클리핑
        x1 = max(0, min(image_width - 1, x1));
        y1 = max(0, min(image_height - 1, y1));
        x2 = max(0, min(image_width - 1, x2));
        y2 = max(0, min(image_height - 1, y2));

        // 디버깅: 계산된 좌표 출력
        cout << "Calculated Coordinates: x1=" << x1
                  << ", y1=" << y1
                  << ", x2=" << x2
                  << ", y2=" << y2 << endl;

        // 바운딩 박스 그리기
        rectangle(img, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 2);
        if (x2 <= x1 || y2 <= y1) {
            cerr << "Invalid box size: width=" << (x2 - x1)
                      << ", height=" << (y2 - y1) << endl;
            return img;

        }

        // 라벨 그리기
        ostringstream label;
        label << "Class " << box.class_id << ": " << fixed << setprecision(2) << box.confidence;
        putText(img, label.str(), Point(x1, y1 - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
    }

    return img;
}


// uint8 -> float 변환 함수
void ConvertUint8ToFloat(const uint8_t* uint8_data, float* float_data, int length, float scale, int zero_point) {
    for (int i = 0; i < length; ++i) {
        float_data[i] = (static_cast<int>(uint8_data[i]) - zero_point) * scale;
    }
}

// float -> uint8 양자화 함수
void QuantizeInput(const float* float_data, uint8_t* quantized_data, int length, float scale, int zero_point) {
    for (int i = 0; i < length; ++i) {
        quantized_data[i] = static_cast<uint8_t>(std::round(float_data[i] / scale) + zero_point);
    }
}

int main(int argc, char* argv[]) {
  const char* filename;
  int CAMSIZE;

  if (argc < 3) {
	filename = "models/b160-int8_edgetpu.tflite";
	CAMSIZE = 160;
  } else {
	filename = argv[1];
	CAMSIZE = atoi(argv[2]);
  }

  if (wiringPiSetup () == -1) {
	exit(EXIT_FAILURE);
  }
  if (setuid(getuid()) < 0) {
    perror("Dropping privileges failed.\n");
    exit(EXIT_FAILURE);
  }

  // (1) Pycam setting
  VideoCapture cap("/dev/video0");
  // 카메라 연결 확인

  if (!cap.isOpened()) {
    cerr << "Error: Unable to open the camera" << endl;
    return -1;
  }

  // 해상도 설정 (선택 사항)
  cap.set(CAP_PROP_FRAME_WIDTH, CAMSIZE);
  cap.set(CAP_PROP_FRAME_HEIGHT, CAMSIZE);

  cout << "Press 'q' to exit the video stream." << endl;
  printf("비디오 세팅 완료!\n");
  // (2) load model
  unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // (3) build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // (4) setup for edge tpu device.
  size_t num_devices;
  unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(
      edgetpu_list_devices(&num_devices), &edgetpu_free_devices);

  assert(num_devices > 0);
  const auto& device = devices.get()[0];

  // (5) create tpu delegate.
  auto* delegate =
    edgetpu_create_delegate(device.type, device.path, nullptr, 0);

  // (6) delegate graph.
  interpreter->ModifyGraphWithDelegate(delegate);

  interpreter->AllocateTensors();

  // 입력 텐서 정보 가져오기
  const TfLiteTensor* input_tensor = interpreter->input_tensor(0);
  float scale = input_tensor->params.scale;
  int zero_point = input_tensor->params.zero_point;
  int input_length = input_tensor->bytes / sizeof(uint8_t);

  Mat image;
  bool detected = true;
  float distance = -1;
  
  while (1) {
	distance = get_distance();
	if (distance > 0 && distance <= 10)
		detected = true;
	
	// 만약 사거리 안에 물체가 감지되는 경우에만 진행
	if (!detected)
		continue;
	cap.read(image);

    // 화면에 프레임 표시
    printf("카메라에서 이미지 불러옴\n");
    cvtColor(image, image, COLOR_BGR2RGB);

    if (image.empty()) {
      cerr << "Error capturing image" << endl;
      break;
    }

    try {
      resize(image, image, Size(CAMSIZE, CAMSIZE), 0, 0, INTER_LINEAR);
    } catch (const cv::Exception& e) {
      cerr << "OpenCV Exception: " << e.what() << "\n";
      return -1;
    }
    printf("이미지 로드!\n");

    // Push image to input tensor
    // 입력 텐서를 uint8_t 포인터로 가져오기
    auto input_uint8 = interpreter->typed_input_tensor<uint8_t>(0);

	// float 데이터를 uint8_t로 변환
    for (int i=0; i<CAMSIZE; i++){
      for (int j=0; j<CAMSIZE; j++){
        cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
        for (int k=0; k<3; k++) {
		  float result = pixel[k] / 255.0;
          input_uint8[(i*CAMSIZE+j)*3+k] = static_cast<uint8_t>(round(result/scale)+zero_point);
		}
      }
	}

    cv::Mat float_image;
    // 픽셀 값 정규화 (0~255 → 0~1)
    image.convertTo(float_image, CV_32F, 1.0 / 255.0);

    // 양자화
    for (int i = 0; i < image.rows; i++) {
      for (int j = 0; j < image.cols; j++) {
        cv::Vec3f pixel = float_image.at<cv::Vec3f>(i, j);
		for (int k = 0; k < 3; k++)
          input_uint8[(i * image.cols + j) * 3 + k] = 
			  static_cast<uint8_t>(std::round(pixel[k] / scale) + zero_point);
      }
	}
    printf("인풋텐서 생성 완료!\n");

    // Run inference
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
    printf("\n\n=== Post-invoke Interpreter State ===\n");
    printf("추론 실행!\n");

    // Output parsing
    const TfLiteTensor* output_tensor = interpreter->output_tensor(0);
    float output_scale = output_tensor->params.scale;
    int output_zero_point = output_tensor->params.zero_point;
    int output_length = output_tensor->bytes / sizeof(uint8_t);

    const uint8_t* uint8_output = interpreter->typed_output_tensor<uint8_t>(0);

    // 출력 데이터를 변환할 float 배열 동적 할당
    float* output_float = static_cast<float*>(malloc(output_length * sizeof(float)));
    if (!output_float) {
      std::cerr << "Failed to allocate memory for float_output." << std::endl;
      free(input_uint8);
      return -1;
    }

    ConvertUint8ToFloat(uint8_output, output_float, output_length, output_scale, output_zero_point);

    // 출력 텐서 크기 정보
    int num_detections = output_tensor->dims->data[1];
    int num_classes = output_tensor->dims->data[2] - 5;

    // YOLO 디코딩
    float conf_threshold = 0.4;
    float iou_threshold = 0.2;

    std::vector<BoundingBox> results = decode_predictions(output_float, num_detections, num_classes, conf_threshold, iou_threshold, CAMSIZE, CAMSIZE);

    // (12) Output visualize
    image = draw_detections(image, results, CAMSIZE, CAMSIZE);

    cv::imshow("Yolo example with Pycam", image);

	// 메모리 해제
    free(output_float);
    char key = cv::waitKey(1);
    if (key == 'q') {
        break;
    }

	detected = false;
  }

  // (13) release
  cap.release();
  destroyAllWindows();

  return 0;
}

