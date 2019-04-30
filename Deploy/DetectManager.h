#pragma once
#include <torch/torch.h> 
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
class Bbox {
private:
	int center_x;
	int center_y;
	int w;
	int h;
public:
	Bbox()
	{
		this->center_x = 0;
		this->center_y = 0;
		this->w = 0;
		this->h = 0;
	}
	void Bbox_set(int center_x, int center_y, int w, int h)
	{
		this->center_x = center_x;
		this->center_y = center_y;
		this->w = w;
		this->h = h;
	}
	int get_center_x()
	{
		return this->center_x;
	}
	int get_center_y() {
		return this->center_y;
	}
	int get_w() {
		return this->w;
	}

	int get_h() {
		return this->h;
	}
	int get_left_border()	{
		return this->center_x - this->w / 2;
	}
	int get_top_border() {
		return this->center_y + this->h / 2;
	}
	int get_right_border()
	{
		return this->center_x - this->w / 2 + this->w;
	}
	int get_bottom_border()
	{
		return this->center_y + this->h / 2 - this->h;
	}
};


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
	void get_target_loc(cv::Mat &img,bool &result,Bbox &target);

};