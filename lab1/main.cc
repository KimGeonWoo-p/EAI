// 추론을 위한 라이브러리
#include <opencv2/dnn.hpp>

// 카메라를 위한 라이브러리
#include <opencv2/opencv.hpp>

#include <iostream>
#include <cstdio>
#include <vector>

// camsize
#define CAMSIZE 320

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

// YOLO 출력 파싱 및 시각화 함수
void drawDetections(cv::Mat& frame, const cv::Mat& output, float confThreshold, float nmsThreshold, const std::vector<std::string>& classNames) {
  std::vector<int> classIds;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  for (int i = 0; i < output.rows; ++i) {
    float confidence = output.at<float>(i, 4);  // Confidence score
    if (confidence > confThreshold) {
      // 바운딩 박스 정보
      float x_center = output.at<float>(i, 0) * frame.cols;
      float y_center = output.at<float>(i, 1) * frame.rows;
      float width = output.at<float>(i, 2) * frame.cols;
      float height = output.at<float>(i, 3) * frame.rows;

      int x = static_cast<int>(x_center - width / 2);
      int y = static_cast<int>(y_center - height / 2);
      boxes.emplace_back(cv::Rect(x, y, static_cast<int>(width), static_cast<int>(height)));

      // 클래스 ID
      cv::Mat scores = output.row(i).colRange(5, output.cols);
      cv::Point classIdPoint;
      double maxClassScore;
      cv::minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint);
      classIds.push_back(classIdPoint.x);
      confidences.push_back(static_cast<float>(maxClassScore));
    }
  }

  // NMS (Non-Max Suppression)로 중복 바운딩 박스 제거
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

  // 바운딩 박스와 클래스 라벨을 이미지에 표시
  for (int idx : indices) {
    cv::Rect box = boxes[idx];
    cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);

    std::string label = classNames[classIds[idx]] + ": " + cv::format("%.2f", confidences[idx]);
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    int top = std::max(box.y, labelSize.height);
    cv::rectangle(frame, cv::Point(box.x, top - labelSize.height),
              cv::Point(box.x + labelSize.width, top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(frame, label, cv::Point(box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
  }
}

int main() {
  if (argc != 2) {
    fprintf(stderr, "minimal <tflite model>\n");
    return 1;
  }
  const char* filename = argv[1];
  
  // TFLite 모델 경로
  std::string model_path = filename;
    
  // TFLite 모델 로드
  cv::dnn::Net net = cv::dnn::readNetFromTensorflow(model_path);
  
  // 카메라 세팅
  cv::VideoCapture video(0);
  if (!video.isOpened())
  {
    cout << "Unable to get video from the camera!" << endl;
    return -1;
  }

  video.set(cv::CAP_PROP_FRAME_WIDTH, CAMSIZE);
  video.set(cv::CAP_PROP_FRAME_HEIGHT, CAMSIZE);

  // 카메라에서 사진을 송출받음
  cv::Mat frame;
  while (video.read(frame)) {
    // 이미지 크기 변경 (모델의 입력 크기에 맞게)
    cv::Mat blob;
    cv::Size input_size(320, 320);  // 모델 입력 크기
    cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, input_size, cv::Scalar(), true, false);

    // 네트워크에 입력 설정
    net.setInput(blob);

    // 추론 수행
    cv::Mat output = net.forward();

    // 시각화
    float confThreshold = 0.5;  // Confidence 임계값
    float nmsThreshold = 0.4;   // NMS 임계값
    drawDetections(frame, output, confThreshold, nmsThreshold, classNames);

    // 결과 이미지 출력
    cv::imshow("Detections", frame);
    cv::waitKey(0);
  }  
  return 0;
}

