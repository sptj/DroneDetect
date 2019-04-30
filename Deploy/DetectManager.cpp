#include <torch/script.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <time.h>
#include "DetectManager.h"
#include "get_anchors.h"
#include <torch/torch.h> 

DetectManager::DetectManager(const int w, const int h)
{
	this->w = w;
	this->h = h;
	anchor_table = get_anchors(w, h);
	std::string model_name = "D:\\DroneDetect\\sptj\\model\\model-resnet50.pt";
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

clock_t start, a, b, end;
double total_time, opencv_cost, pre_process, cuda_infer;

void DetectManager::get_target_loc(cv::Mat &img, bool &result, Bbox &target)
{

	//cv::cvtColor(img, img, CV_BGR2RGB);
	//img.convertTo(img, CV_32F, 1.0 / 255);
	if (img.rows != this->h || img.cols != this->w)
	{
		printf("I detect that the image input is not 1920*1080\n");
		printf("which will pull down the speed of this program\n");
		getchar();
		exit(0);
	}

	auto options = torch::TensorOptions().dtype(torch::kU8);
	torch::Tensor img_tensor = torch::from_blob(img.ptr<unsigned char>(), { 1, this->h, this->w, 3 }, options).to(torch::kCUDA);

	img_tensor = img_tensor.permute({ 0,3,1,2 });
	img_tensor = img_tensor.toType(torch::kFloat32);
	img_tensor.div_(255.0);
	img_tensor[0][0].sub_(0.485).div_(0.229);
	img_tensor[0][1].sub_(0.456).div_(0.224);
	img_tensor[0][2].sub_(0.406).div_(0.225);
	//start = clock();

	//// Execute the model and turn its output into a tensor.	
	torch::Tensor out_tensor = model->forward({ img_tensor }).toTensor();
	result = torch::sigmoid(out_tensor.max()).to(at::kCPU).item().toFloat() > 0.5;
	//std::cout << result << std::endl;
	if (result)
	{
		target.Bbox_set(
			anchor_table[out_tensor.argmax()][0].item().toFloat(),
			anchor_table[out_tensor.argmax()][1].item().toFloat(),
			anchor_table[out_tensor.argmax()][2].item().toFloat(),
			anchor_table[out_tensor.argmax()][3].item().toFloat()
		);		
		//std::cout << anchor_table[out_tensor.argmax()] << std::endl;
		//printf("detected\n");
	}
	else
	{
		target.Bbox_set(0, 0, 0, 0);
	}


	/*end = clock();

	total_time = (double)(end - start) / CLOCKS_PER_SEC;
	printf("----------------------------------------\n");
	printf("total costs %f seconds\n", total_time);
	printf("\n");*/
}























