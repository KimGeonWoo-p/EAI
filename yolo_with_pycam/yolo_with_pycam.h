#include "yolo_parser.h"
yolo::YOLO_Parser yolo_parser;
std::vector<yolo::YOLO_Parser::BoundingBox> yolo::YOLO_Parser::result_boxes;
std::vector<std::vector<float>> yolo::YOLO_Parser::real_bbox_cls_vector; 
std::vector<int> yolo::YOLO_Parser::real_bbox_cls_index_vector;
std::vector<std::vector<int>> yolo::YOLO_Parser::real_bbox_loc_vector;

void yolo_output_parsing(TfLiteTensor* cls_tensor, TfLiteTensor* loc_tensor){
  printf("\033[0;33mStart YOLO parsing\033[0m\n");
  std::vector<int> real_bbox_index_vector;
  real_bbox_index_vector.clear();
  yolo::YOLO_Parser::real_bbox_cls_index_vector.clear();
  yolo::YOLO_Parser::real_bbox_cls_vector.clear();
  yolo::YOLO_Parser::real_bbox_loc_vector.clear();
  yolo::YOLO_Parser::result_boxes.clear();
  yolo_parser.make_real_bbox_cls_vector(cls_tensor, real_bbox_index_vector,
                                         yolo::YOLO_Parser::real_bbox_cls_vector);
  yolo::YOLO_Parser::real_bbox_cls_index_vector = \
              yolo_parser.get_cls_index(yolo::YOLO_Parser::real_bbox_cls_vector); 
  yolo_parser.make_real_bbox_loc_vector(loc_tensor, real_bbox_index_vector, 
                                        yolo::YOLO_Parser::real_bbox_loc_vector);
  float iou_threshold = 0.5;
  yolo_parser.PerformNMSUsingResults(real_bbox_index_vector, yolo::YOLO_Parser::real_bbox_cls_vector, 
        yolo::YOLO_Parser::real_bbox_loc_vector, iou_threshold,yolo::YOLO_Parser::real_bbox_cls_index_vector);
  printf("\033[0;33mEND YOLO parsing\033[0m\n");
};

void yolo_output_visualize(cv::Mat image){
  std::map<int, std::string> labelDict = {
        {0, "person"},     {1, "bicycle"},   {2, "car"},          {3, "motorbike"},
        {4, "aeroplane"},  {5, "bus"},       {6, "train"},        {7, "truck"},
        {8, "boat"},       {9, "traffic_light"}, {10, "fire_hydrant"}, {11, "stop_sign"},
        {12, "parking_meter"}, {13, "bench"}, {14, "bird"},       {15, "cat"},
        {16, "dog"},       {17, "horse"},    {18, "sheep"},       {19, "cow"},
        {20, "elephant"},  {21, "bear"},     {22, "zebra"},       {23, "giraffe"},
        {24, "backpack"},  {25, "umbrella"}, {26, "handbag"},     {27, "tie"},
        {28, "suitcase"},  {29, "frisbee"},  {30, "skis"},        {31, "snowboard"},
        {32, "sports_ball"}, {33, "kite"},   {34, "baseball_bat"}, {35, "baseball_glove"},
        {36, "skateboard"}, {37, "surfboard"}, {38, "tennis_racket"}, {39, "bottle"},
        {40, "wine_glass"}, {41, "cup"},     {42, "fork"},        {43, "knife"},
        {44, "spoon"},     {45, "bowl"},    {46, "banana"},      {47, "apple"},
        {48, "sandwich"},  {49, "orange"},  {50, "broccoli"},    {51, "carrot"},
        {52, "hot_dog"},   {53, "pizza"},   {54, "donut"},       {55, "cake"},
        {56, "chair"},     {57, "sofa"},    {58, "potted_plant"}, {59, "bed"},
        {60, "dining_table"}, {61, "toilet"}, {62, "tvmonitor"}, {63, "laptop"},
        {64, "mouse"},     {65, "remote"},  {66, "keyboard"},    {67, "cell_phone"},
        {68, "microwave"}, {69, "oven"},    {70, "toaster"},     {71, "sink"},
        {72, "refrigerator"}, {73, "book"}, {74, "clock"},       {75, "vase"},
        {76, "scissors"},  {77, "teddy_bear"}, {78, "hair_drier"}, {79, "toothbrush"}
  };
  std::vector<yolo::YOLO_Parser::BoundingBox> bboxes = yolo::YOLO_Parser::result_boxes;
  if (!image.empty()) {
  	    yolo_parser.visualize_with_labels(image, bboxes, labelDict);
        cv::imshow("Yolo example with Pycam", image);
	}
};
