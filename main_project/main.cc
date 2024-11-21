#include <opencv2/opencv.hpp>
#include <fstream>
#include <raspicam/raspicam_cv.h>
#include <vector>

// Namespaces.
using namespace cv;
using namespace std;
using namespace cv::dnn;

// Constants.
const float INPUT_WIDTH = 160.0;
const float INPUT_HEIGHT = 160.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;
 
// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;
 
// Colors.
Scalar BLACK = Scalar(0,0,0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0,0,255);

void draw_label(Mat& input_image, string label, int left, int top)
{
    // Display the label at the top of the bounding box.
    int baseLine;
    Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = max(top, label_size.height);
    // Top left corner.
    Point tlc = Point(left, top);
    // Bottom right corner.
    Point brc = Point(left + label_size.width, top + label_size.height + baseLine);
    // Draw white rectangle.
    rectangle(input_image, tlc, brc, BLACK, FILLED);
    // Put the label on the black rectangle.
    putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

vector<Mat> pre_process(Mat &input_image, Net &net)
{
    // Convert to blob.
    Mat blob;
    blobFromImage(input_image, blob, 1./255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);
 
    net.setInput(blob);
 
    // Forward propagate.
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
 
    return outputs;
}

Mat post_process(Mat &input_image, vector<Mat> &outputs, const vector<string> &class_name, int rows, int demensions)
{
    // Initialize vectors to hold respective outputs while unwrapping     detections.
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;

    // Resizing factor.
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    float *data = (float *)outputs[0].data;

    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            float * classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire the index of best class  score.
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > SCORE_THRESHOLD)
            {
                // Store class ID and confidence in the pre-defined respective vectors.
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);
                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // Store good detections in the boxes vector.
                boxes.push_back(Rect(left, top, width, height));
            }
        }
        // Jump to the next row.
        data += demensions;
    }

    // Perform Non-Maximum Suppression and draw predictions.
    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // Draw bounding box.
        rectangle(input_image, Point(left, top), Point(left + width, top + height), BLUE, 3*THICKNESS);
        // Get the label for the class name and its confidence.
        string label = format("%.2f", confidences[idx]);
        label = class_name[class_ids[idx]] + ":" + label;
        // Draw class labels.
        draw_label(input_image, label, left, top);
    }
    return input_image;
}

int main()
{
    int CAMSIZE = 160;

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

    // (2) Load model
    Net net;
    net = readNet("best-fp16.tflite");

    // (3) Build interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);
    printf("인터프리터 빌드 완료\n");

    // Load class list.
    vector<string> class_list;
    ifstream ifs("coco.names");
    string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }

    int rows = 3*(CAMSIZE*CAMSIZE+(CAMSIZE/2)*(CAMSIZE/2)+(CAMSIZE/4)*(CAMSIZE/4))
    int demensions = class_list.size();

    while (1) {
        // (2) Load image from Pycam
        cv::Mat image;
        camera.grab();
        camera.retrieve(image);
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        printf("카메라에서 이미지 불러옴\n");
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

        vector<Mat> detections;     // Process the image.
        detections = pre_process(image, net);
        Mat img = post_process(image.clone(), detections, class_list, rows, demensions);

        // Put efficiency information.
        // The function getPerfProfile returns the overall time for nference(t) and the timings for each of the layers(in layersTimes).
        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        string label = format("Inference time : %.2f ms", t);
        putText(img, label, Point(20, 40), FONT_FACE, FONT_SCALE, RED);
        imshow("Output", img);
        waitKey(0);
    }
    return 0;
}