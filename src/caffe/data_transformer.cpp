#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV

#include <string>
#include <vector>
#include <sstream>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>

#define SSTR( x ) static_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()

namespace caffe {

template<typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param,
    Phase phase)
    : param_(param), phase_(phase) {
  // check if we want to use mean_file
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0) <<
      "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = param.mean_file();
    if (Caffe::root_solver()) {
      LOG(INFO) << "Loading mean file from: " << mean_file;
    }
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(param_.has_mean_file() == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  int crop_h = 0;
  int crop_w = 0;
  if (param_.has_crop_size()) {
    crop_h = param_.crop_size();
    crop_w = param_.crop_size();
  }
  if (param_.has_crop_h()) {
    crop_h = param_.crop_h();
  }
  if (param_.has_crop_w()) {
    crop_w = param_.crop_w();
  }
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_h);
  CHECK_GE(datum_width, crop_w);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
     "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_h || crop_w) {
    height = crop_h;
    width = crop_w;
    // We only do random crop when we do training.
    if (phase_ == TRAIN && !param_.center_crop()) {
      h_off = Rand(datum_height - crop_h + 1);
      w_off = Rand(datum_width - crop_w + 1);
    } else {
      h_off = (datum_height - crop_h) / 2;
      w_off = (datum_width - crop_w) / 2;
    }
  }

  cv::RNG rng(caffe_rng_rand());//caffe's rng is difficult to use.
  const bool do_erase = param_.has_erase_ratio() & (rng.uniform(0.0,1.0) < param_.erase_ratio());
  int erase_x_min = width, erase_x_max = -1, erase_y_min = height, erase_y_max = -1;
  if (do_erase) {
    do {
      float erase_scale = rng.uniform(param_.scale_min(), param_.scale_max());
      int erase_width = (float)width * erase_scale;
      float erase_aspect = rng.uniform(param_.aspect_min(), param_.aspect_max());
      int erase_height = (float)erase_width * erase_aspect;
      erase_x_min = rng.uniform(0, width);
      erase_y_min = rng.uniform(0, height);
      erase_x_max = erase_x_min + erase_width - 1;
      erase_y_max = erase_y_min + erase_height - 1;
    } while (erase_x_min < 0 || erase_y_min < 0 || erase_x_max >= width || erase_y_max >= height);
  }

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (do_erase && w >= erase_x_min && w <= erase_x_max && h >= erase_y_min && h <= erase_y_max) {
          datum_element = Rand(255);
        }
        else {
          if (has_uint8) {
            datum_element =
              static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
          }
          else {
            datum_element = datum.float_data(data_index);
          }
        }
        if (has_mean_file) {
          transformed_data[top_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
      }
    }
  }
}


template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Blob<Dtype>* transformed_blob) {
  // If datum is encoded, decode and transform the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Transform the cv::image into blob.
    return Transform(cv_img, transformed_blob);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  int crop_h = 0;
  int crop_w = 0;
  if (param_.has_crop_size()) {
    crop_h = param_.crop_size();
    crop_w = param_.crop_size();
  }
  if (param_.has_crop_h()) {
    crop_h = param_.crop_h();
  }
  if (param_.has_crop_w()) {
    crop_w = param_.crop_w();
  }
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, datum_channels);
  CHECK_LE(height, datum_height);
  CHECK_LE(width, datum_width);
  CHECK_GE(num, 1);

  if (crop_h || crop_w) {
    CHECK_EQ(crop_h, height);
    CHECK_EQ(crop_w, width);
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  Transform(datum, transformed_data);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum> & datum_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_LE(datum_num, num) <<
    "The size of datum_vector must be no greater than transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
}

#ifdef USE_OPENCV
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat> & mat_vector,
  Blob<Dtype>* transformed_blob,
  bool transpose) {
  const int mat_num = mat_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  
  CHECK_GT(mat_num, 0) << "There is no MAT to add";
  CHECK_EQ(mat_num, num) <<
    "The size of mat_vector must be equals to transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < mat_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(mat_vector[item_id], &uni_blob, transpose);
  }
}

// template<typename Dtype>
// Dtype DataTransformer<Dtype>::randDouble() {
//   rng_t* rng =
//       static_cast<rng_t*>(rng_->generator());
//   uint64_t randval = (*rng)();

//   return (Dtype(randval) / Dtype(rng_t::max()));
// }

static int debug_count = 0;

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
  Blob<Dtype>* transformed_blob,
  bool transpose) {
  int crop_h = 0;
  int crop_w = 0;
  if (param_.has_crop_size()) {
    crop_h = param_.crop_size();
    crop_w = param_.crop_size();
  }
  if (param_.has_crop_h()) {
    crop_h = param_.crop_h();
  }
  if (param_.has_crop_w()) {
    crop_w = param_.crop_w();
  }
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, img_channels);
  if (transpose) {
    CHECK_LE(height, img_width);
    CHECK_LE(width, img_height);
  }
  else {
    CHECK_LE(height, img_height);
    CHECK_LE(width, img_width);
  }
  CHECK_GE(num, 1);
  //if (transpose) {
  //  std::swap(height, width);
  //}

  //CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  //////////////////////
  // custom augmentation before crop
  //////////////////////

  const string debug_path = param_.debug_path();
  const bool debug = debug_path.length() > 0;

  // rotate and scale
  const int rotation_angle_max_min = param_.rotation_angle();

  const int min_scaling_factor = param_.min_scaling_factor();
  const int max_scaling_factor = param_.max_scaling_factor();

  bool doScale = min_scaling_factor != 100 || max_scaling_factor != 100;
  bool doRotate = rotation_angle_max_min != 0;

  if (doScale || doRotate) {
    
    int rotation_degree = 0;
    if (doRotate) {
      rotation_degree = Rand(rotation_angle_max_min+1);
      if(Rand(2) == 0) {
        rotation_degree = -rotation_degree;
      }
    }

    float scale = 1.0;

    if(doScale) {
      int scale_diff = max_scaling_factor-min_scaling_factor;
      int scale_shift = Rand(scale_diff+1);

      scale = (min_scaling_factor + scale_shift)*0.01;
    }
    
    cv::Mat dst;
    cv::Point2f pt(cv_img.cols/2., cv_img.rows/2.);    
    cv::Mat r = getRotationMatrix2D(pt, rotation_degree, scale);
    warpAffine(cv_img, dst, r, cv::Size(cv_img.cols, cv_img.rows), cv::INTER_CUBIC);

    dst.copyTo(cv_img);
  }


  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_h);
  CHECK_GE(img_width, crop_w);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
      "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img;
  if (crop_h || crop_w) {
    CHECK_EQ(crop_h, height);
    CHECK_EQ(crop_w, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN && !param_.center_crop()) {
      h_off = Rand(img_height - crop_h + 1);
      w_off = Rand(img_width - crop_w + 1);
    }
    else {
      h_off = (img_height - crop_h) / 2;
      w_off = (img_width - crop_w) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_w, crop_h);
    cv_cropped_img = cv_img(roi);
  }
  else {
    if (transpose) {
      CHECK_EQ(img_width, height);
      CHECK_EQ(img_height, width);
    }
    else {
      CHECK_EQ(img_height, height);
      CHECK_EQ(img_width, width);
    }
  }

  cv::RNG rng(caffe_rng_rand());//caffe's rng is difficult to use.
  const bool do_erase = param_.has_erase_ratio() & (rng.uniform(0.0, 1.0) < param_.erase_ratio());
  int erase_x_min = width, erase_x_max = -1, erase_y_min = height, erase_y_max = -1;
  if (do_erase) {
    do {
      float erase_scale = rng.uniform(param_.scale_min(), param_.scale_max());
      int erase_width = (float)width * erase_scale;
      float erase_aspect = rng.uniform(param_.aspect_min(), param_.aspect_max());
      int erase_height = (float)erase_width * erase_aspect;
      erase_x_min = rng.uniform(0, width);
      erase_y_min = rng.uniform(0, height);
      erase_x_max = erase_x_min + erase_width - 1;
      erase_y_max = erase_y_min + erase_height - 1;
    } while (erase_x_min < 0 || erase_y_min < 0 || erase_x_max >= width || erase_y_max >= height);
  }

  CHECK(cv_cropped_img.data);

  //////////////////////
  // custom augmentation after crop
  //////////////////////

  // optional uint32 contrast_adjustment_min_alpha = 20 [default = 100];
  // optional uint32 contrast_adjustment_max_alpha = 21 [default = 100];
  // optional uint32 contrast_adjustment_beta = 22 [default = 0];
  // optional bool contrast_adjustment_prob = 23 [default = 100]; // 100 - apply to each image


  // Contrast and Brightness Adjuestment
  const int contrast_min_alpha = param_.contrast_min_alpha();
  const int contrast_max_alpha = param_.contrast_max_alpha();
  const int contrast_min_beta = param_.contrast_min_beta();
  const int contrast_max_beta = param_.contrast_max_beta();
  const int contrast_prob = param_.contrast_prob();
  
  bool doContrastAdjustment = contrast_prob > 0 && 
        (contrast_min_alpha != 100 || contrast_max_alpha != 100 || 
          contrast_min_beta != 0 || contrast_max_beta != 0);

  if(doContrastAdjustment) {
    float alpha = 1.0, beta = 0.0;
    int apply_contrast = Rand(100);
    if(apply_contrast < contrast_prob) {
        int alpha_shift = Rand(contrast_max_alpha - contrast_min_alpha + 1);
        alpha = (contrast_min_alpha + alpha_shift)/100.0;
        int beta_shift = Rand(contrast_max_beta - contrast_min_beta + 1);
        beta = (contrast_min_beta + beta_shift)/100.0;
        // printf("!!!! %f, %f\n", alpha, beta);
        cv_cropped_img.convertTo(cv_cropped_img, -1 , alpha, beta);
    }
  }  

  // hue&saturation
  if(img_channels == 3) {
    const int saturate_min = param_.saturate_min();
    const int saturate_max = param_.saturate_max();
    const int saturate_prob = param_.saturate_prob();

    const int hue_rotation_min = param_.hue_rotation_min();
    const int hue_rotation_max = param_.hue_rotation_max();
    const int hue_rotation_prob = param_.hue_rotation_prob();

    if(saturate_prob > 0 || hue_rotation_prob > 0) {

      float saturate = 1.0;
      float hue_rotation = 0.0;

      if(saturate_prob > 0) {
        int apply_saturate = Rand(100);
        if(apply_saturate < saturate_prob) {
          int saturate_rnd = Rand(saturate_max - saturate_min + 1);
          saturate = (saturate_min + saturate_rnd)/100.0;        
        }
      }

      if(hue_rotation_prob > 0) {
        int apply_hue_rotation = Rand(100);
        if(apply_hue_rotation < hue_rotation_prob) {
          int hue_rotation_rnd = Rand(hue_rotation_max - hue_rotation_min + 1);
          hue_rotation = (hue_rotation_min + hue_rotation_rnd)/100.0;
        }
      }



      if (saturate != 1.0 || hue_rotation != 0.0) {

        cv_cropped_img.convertTo(cv_cropped_img, CV_32FC3);

        cvtColor(cv_cropped_img, cv_cropped_img, CV_BGR2HSV);

        vector<cv::Mat1f> channels(3);
        cv::split(cv_cropped_img, channels);
        if (hue_rotation != 0.0) {
          channels[0] += hue_rotation;
        }
        if (saturate != 1.0) {
          channels[1] *= saturate;
        }
    
        cv::merge(channels, cv_cropped_img);

        cvtColor(cv_cropped_img, cv_cropped_img, CV_HSV2BGR);
      }
    }
  }

  // Smooth
  const int smooth_filtering_prob = param_.smooth_filtering_prob();
  const google::protobuf::RepeatedField<unsigned int> smooth_filtering = param_.smooth_filtering();

  if(smooth_filtering_prob > 0 && smooth_filtering.size() > 0) {
    int apply_smooth = Rand(100);
    if(apply_smooth < smooth_filtering_prob) {
      int kernel_idx = Rand(smooth_filtering.size());
      int kernel = smooth_filtering.Get(kernel_idx);
      cv::GaussianBlur(cv_cropped_img, cv_cropped_img, cv::Size(kernel,kernel),0);
    }    
  }


  if(debug) {
    imwrite(debug_path + "/" + SSTR(debug_count) + ".png", cv_cropped_img);
    debug_count++;
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  bool is_float_data = cv_cropped_img.depth() == CV_32F;
  int top_index;
  if (transpose) {
    // for (int w = 0; w < width; ++w) {
    for (int h = 0; h < height; ++h) {
      const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
      const float* float_ptr = cv_cropped_img.ptr<float>(h);
      int img_index = 0;
      // for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        for (int c = img_channels - 1; c >= 0; --c) {
          if (do_mirror) {
            top_index = (c * height + h) * width + (width - 1 - w);
          }
          else {
            top_index = (c * height + h) * width + w;
          }
          // int top_index = (c * height + h) * width + w;
          Dtype pixel;
          if (do_erase && w >= erase_x_min && w <= erase_x_max && h >= erase_y_min && h <= erase_y_max) {
            pixel = Rand(255);
          }
          else {
            pixel = static_cast<Dtype>(is_float_data ? float_ptr[img_index] : ptr[img_index]);
          }
          img_index++;
          if (has_mean_file) {
            int mean_index;
            mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
            transformed_data[top_index] =
              (pixel - mean[mean_index]) * scale;
          }
          else {
            if (has_mean_values) {
              transformed_data[top_index] =
                (pixel - mean_values_[c]) * scale;
            }
            else {
              transformed_data[top_index] = pixel * scale;
            }
          }
        }
      }
    } 
  }
  else {
    for (int h = 0; h < height; ++h) {
      const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
      const float* float_ptr = cv_cropped_img.ptr<float>(h);
      int img_index = 0;
      for (int w = 0; w < width; ++w) {
        for (int c = 0; c < img_channels; ++c) {
          if (do_mirror) {
            top_index = (c * height + h) * width + (width - 1 - w);
          }
          else {
            top_index = (c * height + h) * width + w;
          }
          // int top_index = (c * height + h) * width + w;
          Dtype pixel;
          if (do_erase && w >= erase_x_min && w <= erase_x_max && h >= erase_y_min && h <= erase_y_max) {
            pixel = Rand(255);
          }
          else {
            pixel = static_cast<Dtype>(is_float_data ? float_ptr[img_index] : ptr[img_index]);
          }
          img_index++;
          if (has_mean_file) {
            int mean_index;
            mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
            transformed_data[top_index] =
              (pixel - mean[mean_index]) * scale;
          }
          else {
            if (has_mean_values) {
              transformed_data[top_index] =
                (pixel - mean_values_[c]) * scale;
            }
            else {
              transformed_data[top_index] = pixel * scale;
            }
          }
        }
      }
    }
  }
}
#endif  // USE_OPENCV

template<typename Dtype>
void DataTransformer<Dtype>::Transform(Blob<Dtype>* input_blob,
                                       Blob<Dtype>* transformed_blob) {
  int crop_h = 0;
  int crop_w = 0;
  if (param_.has_crop_size()) {
    crop_h = param_.crop_size();
    crop_w = param_.crop_size();
  }
  if (param_.has_crop_h()) {
    crop_h = param_.crop_h();
  }
  if (param_.has_crop_w()) {
    crop_w = param_.crop_w();
  }
  const int input_num = input_blob->num();
  const int input_channels = input_blob->channels();
  const int input_height = input_blob->height();
  const int input_width = input_blob->width();

  if (transformed_blob->count() == 0) {
    // Initialize transformed_blob with the right shape.
    if (crop_h && crop_w) {
      transformed_blob->Reshape(input_num, input_channels,
                                crop_h, crop_w);
    } else {
      transformed_blob->Reshape(input_num, input_channels,
                                input_height, input_width);
    }
  }

  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int size = transformed_blob->count();

  CHECK_LE(input_num, num);
  CHECK_EQ(input_channels, channels);
  CHECK_GE(input_height, height);
  CHECK_GE(input_width, width);


  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  int h_off = 0;
  int w_off = 0;
  if (crop_h && crop_w) {
    CHECK_EQ(crop_h, height);
    CHECK_EQ(crop_w, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN && !param_.center_crop()) {
      h_off = Rand(input_height - crop_h + 1);
      w_off = Rand(input_width - crop_w + 1);
    } else {
      h_off = (input_height - crop_h) / 2;
      w_off = (input_width - crop_w) / 2;
    }
  } else {
    CHECK_EQ(input_height, height);
    CHECK_EQ(input_width, width);
  }

  Dtype* input_data = input_blob->mutable_cpu_data();
  if (has_mean_file) {
    CHECK_EQ(input_channels, data_mean_.channels());
    CHECK_EQ(input_height, data_mean_.height());
    CHECK_EQ(input_width, data_mean_.width());
    for (int n = 0; n < input_num; ++n) {
      int offset = input_blob->offset(n);
      caffe_sub(data_mean_.count(), input_data + offset,
            data_mean_.cpu_data(), input_data + offset);
    }
  }

  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels) <<
     "Specify either 1 mean_value or as many as channels: " << input_channels;
    if (mean_values_.size() == 1) {
      caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
    } else {
      for (int n = 0; n < input_num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
          int offset = input_blob->offset(n, c);
          caffe_add_scalar(input_height * input_width, -(mean_values_[c]),
            input_data + offset);
        }
      }
    }
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();

  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * height;
      int data_index_c = (data_index_n + c) * input_height + h_off;
      for (int h = 0; h < height; ++h) {
        int top_index_h = (top_index_c + h) * width;
        int data_index_h = (data_index_c + h) * input_width + w_off;
        if (do_mirror) {
          int top_index_w = top_index_h + width - 1;
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_w-w] = input_data[data_index_h + w];
          }
        } else {
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_h + w] = input_data[data_index_h + w];
          }
        }
      }
    }
  }
  if (scale != Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_scal(size, scale, transformed_data);
  }
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const Datum& datum) {
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // InferBlobShape using the cv::image.
    return InferBlobShape(cv_img);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  }
  int crop_h = 0;
  int crop_w = 0;
  if (param_.has_crop_size()) {
    crop_h = param_.crop_size();
    crop_w = param_.crop_size();
  }
  if (param_.has_crop_h()) {
    crop_h = param_.crop_h();
  }
  if (param_.has_crop_w()) {
    crop_w = param_.crop_w();
  }
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  // Check dimensions.
  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_h);
  CHECK_GE(datum_width, crop_w);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = datum_channels;
  shape[2] = (crop_h)? crop_h : datum_height;
  shape[3] = (crop_w)? crop_w : datum_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<Datum> & datum_vector) {
  const int num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to in the vector";
  // Use first datum in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(datum_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}

#ifdef USE_OPENCV
template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const cv::Mat& cv_img) {
  int crop_h = 0;
  int crop_w = 0;
  if (param_.has_crop_size()) {
    crop_h = param_.crop_size();
    crop_w = param_.crop_size();
  }
  if (param_.has_crop_h()) {
    crop_h = param_.crop_h();
  }
  if (param_.has_crop_w()) {
    crop_w = param_.crop_w();
  }
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;
  // Check dimensions.
  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_h);
  CHECK_GE(img_width, crop_w);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = img_channels;
  shape[2] = (crop_h)? crop_h : img_height;
  shape[3] = (crop_w)? crop_w : img_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<cv::Mat> & mat_vector) {
  const int num = mat_vector.size();
  CHECK_GT(num, 0) << "There is no cv_img to in the vector";
  // Use first cv_img in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(mat_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}
#endif  // USE_OPENCV

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = param_.mirror() ||
      (phase_ == TRAIN && (param_.crop_size() || param_.crop_h() || param_.crop_w()) && !param_.center_crop()) ||
    param_.has_erase_ratio();
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int DataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
