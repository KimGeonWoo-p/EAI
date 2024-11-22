#include <vector>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <cmath>

// 바운딩 박스 구조체
struct BoundingBox {
    float x, y, width, height;
    float confidence;
    int class_id;
};

// 시그모이드 함수
float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

// YOLO 디코딩 함수
std::vector<BoundingBox> decode_predictions(const float* output, int num_detections, int num_classes, float conf_threshold, float iou_threshold) {
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    for (int i = 0; i < num_detections; ++i) {
        const float* detection = output + i * (num_classes + 5);

        float confidence = detection[4];
        if (confidence > conf_threshold) {
            float x = detection[0];
            float y = detection[1];
            float width = detection[2];
            float height = detection[3];

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
                int left = static_cast<int>(x - width / 2);
                int top = static_cast<int>(y - height / 2);

                boxes.emplace_back(left, top, static_cast<int>(width), static_cast<int>(height));
                confidences.push_back(final_score);
                class_ids.push_back(class_id);
            }
        }
    }

    // OpenCV NMS
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

int main() {
    // Python 데이터
    const float output[] = {
        0.5, 0.5, 0.2, 0.2, 0.9, 0.1, 0.7, 0.2, 0.0, // detection 1
        0.6, 0.6, 0.3, 0.3, 0.8, 0.3, 0.5, 0.4, 0.0, // detection 2
        0.4, 0.4, 0.1, 0.1, 0.2, 0.5, 0.2, 0.1, 0.2  // detection 3
    };

    int num_detections = 3;
    int num_classes = 3;
    float conf_threshold = 0.5;
    float iou_threshold = 0.4;

    std::vector<BoundingBox> results = decode_predictions(output, num_detections, num_classes, conf_threshold, iou_threshold);

    // C++ 결과 출력
    std::cout << "C++ Results:" << std::endl;
    for (const auto& bbox : results) {
        std::cout << "BoundingBox: [x=" << bbox.x << ", y=" << bbox.y
                  << ", width=" << bbox.width << ", height=" << bbox.height
                  << ", confidence=" << bbox.confidence
                  << ", class_id=" << bbox.class_id << "]" << std::endl;
    }

    return 0;
}

