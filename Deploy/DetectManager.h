#pragma once
#include <torch/torch.h> 
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class DetectManager
{
private:
	int w;
	int h;
	torch::Tensor anchor_table;
	std::shared_ptr<torch::jit::script::Module> model;
	std::vector<torch::jit::IValue> inputs;

public:
	DetectManager(int w, int h);
	bool get_target_loc(cv::Mat img);

};