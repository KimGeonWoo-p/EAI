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
        {0, "bird"},     {1, "deer"},   {2, "egretta"},          {3, "hare"},
        {4, "sheep"},  {5, "wild boar"},       {6, "wild cat"}
  };
  std::vector<yolo::YOLO_Parser::BoundingBox> bboxes = yolo::YOLO_Parser::result_boxes;
  if (!image.empty()) {
  	    yolo_parser.visualize_with_labels(image, bboxes, labelDict);
        cv::imshow("Yolo example with Pycam", image);
	}
};
