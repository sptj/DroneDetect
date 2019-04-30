#include <torch/script.h>
#include <torch/torch.h> 
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include"DetectManager.h"
#include<time.h>
const char * example_img_path = "G:\\sptj\\IMG_3321.JPG";
int main()

{
	cv::Mat image;

	if (_access(example_img_path, 06)!=-1)
	{
		printf("example image exists at %s\n", example_img_path);
	}
	else 
	{
		printf("%s is a wrong path. Please check it over.\n", example_img_path);
		getchar();
		exit(0);
	}
	image = cv::imread(example_img_path, 1);
	DetectManager d = DetectManager(1920, 1080);
	int start = time(0);
	//std::cout << start << std::endl;
	for (int i = 0; i < 12; i++)
	{
		d.get_target_loc(image);
	}
	int end = time(0);
	//std::cout << "number is :" << std::endl;
	std::cout <<"total time is" <<end - start << std::endl;
	//cv::imshow("NULL", image);
	//cv::waitKey(0);
	getchar();
	return 0;
}
//using namespace cv;
//using namespace std;
//
//int main()
//{
//	读取视频或摄像头
//	VideoCapture capture;
//	const std::string addr="rtsp://admin:jishukaifa3432@192.168.1.64:554/Streaming/Channels/101?transportmode=unicast";
//	Mat frame;
//	capture.open(addr);
//	if (!capture.open(addr))
//	{
//		cout << "is not opened" << endl;
//	}
//	while (true)
//	{
//		
//		capture >> frame;
//		if (!capture.read(frame))
//		{
//			cout << "is not read" << endl;
//		}
//		if(frame.empty())
//			break;
//		imshow("1", frame);
//		waitKey(30);	//延时30
//	}
//	getchar();
//	return 0;
//}

/* main */
int xmain(int argc, const char* argv[]) {
	char *model_name = "D:\\sptj\\model.pt";
	std::ifstream in(model_name, std::ios_base::binary);

	if (in.fail()) {
		std::cout << "failed to open model" << std::endl;
	}
	else {
		std::cout << "successed to open model" << std::endl;
	}

	AT_CHECK(!in.fail(), "load: could not open file ", model_name);



	// Deserialize the ScriptModule from a file using torch::jit::load().
	std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(in);

	assert(module != nullptr);
	std::cout << "load model ok\n";

	// Create a vector of inputs.
	std::vector<torch::jit::IValue> inputs;
	//inputs.push_back(torch::rand({ 1, 3, 1280, 960 }).to(at::kCUDA));
	module->to(at::kCUDA);

	//at::Tensor out_tensor=module->forward({ inputs }).toTensor();
	//std::cout << out_tensor.slice(/*dim=*/1, /*start=*/0, /*end=*/20) << '\n';
	//std::cout << out_tensor.argmax() << '\n';
	//getchar();
	//// evalute time
	//double t = (double)cv::getTickCount();
	//module->forward(inputs).toTensor();
	//t = (double)cv::getTickCount() - t;
	//printf("execution time = %gs\n", t / cv::getTickFrequency());
	//inputs.pop_back();

	//// load image with opencv and transform
	cv::Mat image;
	image = cv::imread("D:\\sptj\\IMG_3321.JPG", 1);
	cv::cvtColor(image, image, CV_BGR2RGB);
	cv::Mat img_float;
	std::cout << image.size << std::endl;
	image.convertTo(img_float, CV_32F, 1.0 / 255);
	cv::resize(img_float, img_float, cv::Size(1280, 960));
	////std::cout << img_float.at<cv::Vec3f>(56,34)[1] << std::endl;
	//std::cout << img_float.size << std::endl;

	auto img_tensor = torch::CPU(torch::kFloat32).tensorFromBlob(img_float.data, { 1, 960, 1280, 3 });

	img_tensor = img_tensor.permute({ 0,3,1,2 });
	img_tensor[0][0] = img_tensor[0][0].sub_(0.485).div_(0.229);
	img_tensor[0][1] = img_tensor[0][1].sub_(0.456).div_(0.224);
	img_tensor[0][2] = img_tensor[0][2].sub_(0.406).div_(0.225);
	auto img_var = torch::autograd::make_variable(img_tensor, false);
	inputs.push_back(img_var.to(at::kCUDA));
	//// Execute the model and turn its output into a tensor.
	torch::Tensor out_tensor = module->forward(inputs).toTensor();
	std::cout << out_tensor.argmax() << std::endl;
	getchar();
}
//	getchar();
//	//// Load labels
//	//std::string label_file = argv[3];
//	//std::ifstream rf(label_file.c_str());
//	//CHECK(rf) << "Unable to open labels file " << label_file;
//	//std::string line;
//	//std::vector<std::string> labels;
//	//while (std::getline(rf, line))
//	//	labels.push_back(line);
//
//	//// print predicted top-5 labels
//	//std::tuple<torch::Tensor, torch::Tensor> result = out_tensor.sort(-1, true);
//	//torch::Tensor top_scores = std::get<0>(result)[0];
//	//torch::Tensor top_idxs = std::get<1>(result)[0].toType(torch::kInt32);
//
//	//auto top_scores_a = top_scores.accessor<float, 1>();
//	//auto top_idxs_a = top_idxs.accessor<int, 1>();
//
//	//for (int i = 0; i < 5; ++i) {
//	//	int idx = top_idxs_a[i];
//	//	std::cout << "top-" << i + 1 << " label: ";
//	//	std::cout << labels[idx] << ", score: " << top_scores_a[i] << std::endl;
//	//}
//
//	return 0;
//}
//
