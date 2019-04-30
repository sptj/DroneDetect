#include <torch/script.h>
#include <torch/torch.h> 
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include<time.h>
#include "DetectManager.h"
#include "get_anchors.h"


DetectManager::DetectManager(const int w,const int h)
{
	this->w = w;
	this->h = h;
	anchor_table = get_anchors(w, h);
	std::string model_name = "G:\\sptj\\model-resnet34.pt";
	if (_access(model_name.c_str(), 06) != -1)
	{
		printf("Use model locate at: %s\n", model_name.c_str());
	}
	else
	{
		printf("%s is a wrong path. Please check it over.\n", model_name.c_str());
		getchar();
		exit(0);
	}

	std::ifstream in(model_name, std::ios_base::binary);
	

	model = torch::jit::load(in);
	assert(model != nullptr);
	std::cout << "load model ok\n";
	model->to(at::kCUDA);
}
clock_t start,a,b,end;
double total_time, opencv_cost, pre_process, cuda_infer;
bool DetectManager::get_target_loc(cv::Mat image)
{
	start = clock();
	cv::cvtColor(image, image, CV_BGR2RGB);
	a = clock();
	cv::Mat img_float;
	image.convertTo(img_float, CV_32F, 1.0 / 255);
	b = clock();
	if (image.rows != this->h || image.cols != this->w)
	{
		printf("I detect that the image input is not 1920*1080\n");
		printf("which will pull down the speed of this program\n");
		cv::resize(img_float, img_float, cv::Size(this->w, this->h));
	}
	end = clock();
	torch::Tensor img_tensor = torch::CPU(torch::kFloat32).tensorFromBlob(img_float.data, { 1, this->h, this->w, 3 });
	//auto img_tensor = torch::CUDA(torch::kFloat32).tensorFromBlob(img_float.data, { 1, 960, 1280, 3 });
	img_tensor = img_tensor.to(at::kCUDA);
	img_tensor = img_tensor.permute({ 0,3,1,2 });
	img_tensor[0][0] = img_tensor[0][0].sub_(0.485).div_(0.229);
	img_tensor[0][1] = img_tensor[0][1].sub_(0.456).div_(0.224);
	img_tensor[0][2] = img_tensor[0][2].sub_(0.406).div_(0.225);
	torch::autograd::Variable img_var = torch::autograd::make_variable(img_tensor, false);
	
	//// Execute the model and turn its output into a tensor.
	
	torch::Tensor out_tensor = model->forward({ img_var }).toTensor();
	
	total_time = (double)(end - start) / CLOCKS_PER_SEC;
	opencv_cost =(double)(a- start) / CLOCKS_PER_SEC;
	pre_process= (double)(b-a) / CLOCKS_PER_SEC;
	cuda_infer = (double)(end- b) / CLOCKS_PER_SEC;
	printf("%f\n", start);
	printf("opencv_cost %f seconds\n", opencv_cost);
	printf("pre_process costs %f seconds\n", pre_process);
	printf("cuda_infer costs %f seconds\n", cuda_infer);
	printf("total costs %f seconds\n", total_time);
	printf("\n");
	if (torch::sigmoid(out_tensor.max()).to(at::kCPU).item().toFloat() > 0.5)
	{	
		//printf("output is %d",out_tensor.argmax());
		//std::cout << anchor_table[out_tensor.argmax()] << std::endl;
		return true;
	}		

	else
	{
		printf("undetected\n");
		return false;
	}
}
