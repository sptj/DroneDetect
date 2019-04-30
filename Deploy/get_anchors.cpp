#include<torch/torch.h>
#include<cmath>
#define anchor_areas_len  5
#define aspect_ratios_len  3
#define scale_ratios_len  3
#define wh_len  2

static  torch::Tensor get_meshgrid(const int w, const int h)
{
	torch::Tensor a = torch::arange(0, w);
	torch::Tensor b = torch::arange(0, h);
	torch::Tensor xx = a.repeat(h).view({ -1, 1 });
	torch::Tensor yy = b.view({ -1, 1 }).repeat({ 1, w }).view({ -1, 1 });
	return torch::cat({ xx, yy }, 1);
}

static  torch::Tensor get_anchors_wh()
{
	torch::Tensor anchor_wh = torch::zeros({ anchor_areas_len,aspect_ratios_len,scale_ratios_len,wh_len });
	float anchor_areas[anchor_areas_len] = { 32 * 32.0,	 64 * 64.0,	128 * 128.0, 256 * 256.0,512 * 512.0 };
	float aspect_ratios[aspect_ratios_len] = { 1 / 2.0, 1 / 1.0, 2 / 1.0 };
	float scale_ratios[scale_ratios_len] = { 1., pow(2, 1 / 3.), pow(2, 2 / 3.) };

	float atom_w, atom_h, anchor_w, anchor_h;
	for (int i = 0; i < anchor_areas_len; i++)
	{
		for (int j = 0; j < aspect_ratios_len; j++)
		{
			atom_h=sqrt(anchor_areas[i]/aspect_ratios[j]);
			atom_w = atom_h*aspect_ratios[j];
			for (int k = 0; k < scale_ratios_len; k++)
			{
				anchor_w = atom_w*scale_ratios[k];
				anchor_h = atom_h*scale_ratios[k];
				anchor_wh[i][j][k][0] = anchor_w;
				anchor_wh[i][j][k][1] = anchor_h;
			}		
		}
	}

	return anchor_wh.view({ anchor_areas_len, -1,2});
}
torch::Tensor get_anchors(const int w,const  int h)
{	
	int fm_w, fm_h;
	torch::Tensor grid_size;
	torch::Tensor xy;
	torch::Tensor wh;
	torch::Tensor boxes[anchor_areas_len];
	for (int i = 0; i < anchor_areas_len; i++)
	{
		fm_w = ceil(w / pow(2, i + 3));
		fm_h = ceil(h / pow(2, i + 3));
		grid_size = torch::tensor({ w / float(fm_w),h / float(fm_h) });
		xy=get_meshgrid(fm_w,fm_h)+0.5;
		xy = (xy*grid_size).view({ fm_h,fm_w,1,2 }).expand({ fm_h,fm_w,9,2 });
		wh = get_anchors_wh()[i].view({ 1, 1, 9, 2 }).expand({ fm_h,fm_w,9,2 });
		boxes[i] = torch::cat({ xy,wh }, 3).view({ -1,4 });
	}
	return torch::cat(boxes, 0);


}

#undef anchor_areas_len  
#undef aspect_ratios_len  
#undef scale_ratios_len  
#undef wh_len  

