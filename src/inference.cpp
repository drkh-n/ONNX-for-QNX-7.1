// // // // // // // // // // // // // //
//                                     //
//                                     //
// ONNX Inference Example on QNX RTOS  //
//                                     //
//                                     //
// // // // // // // // // // // // // //

#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <dirent.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "lodepng.h"

std::vector<float> read_png_as_float(const std::string &filename, unsigned &width, unsigned &height) {
    std::vector<unsigned char> png;
    std::vector<unsigned char> image; // 8-bit RGBA
    unsigned error;

    error = lodepng::load_file(png, filename);
    if (error) throw std::runtime_error("Error loading file " + filename + ": " + lodepng_error_text(error));

    error = lodepng::decode(image, width, height, png);
    if (error) throw std::runtime_error("Error decoding PNG " + filename + ": " + lodepng_error_text(error));

    std::vector<float> result;
    result.reserve(image.size());

    for (size_t i = 0; i < image.size() - 4; i=i+4) {
        result.push_back(image[i] / 255.0f);
    }

    return result;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

void save_grayscale_png_from_float(float* data, size_t width, size_t height, const std::string& filename, float threshold, bool normalize = true) {
    std::vector<unsigned char> image(width * height);

    for (size_t i = 0; i < width * height; ++i) {
        float val = sigmoid(data[i]);

        if (val >= threshold) {
            val = 1.0;
        } else {
            val = 0.0;
        }

        if (normalize)
            val *= 255.0f;
        val = std::max(0.0f, std::min(val, 255.0f));  // manual clamp
        image[i] = static_cast<unsigned char>(val);
    }

    unsigned error = lodepng::encode(filename, image, width, height, LCT_GREY);
    if (error) {
        std::cerr << "PNG encode error " << error << ": " << lodepng_error_text(error) << std::endl;
    } else {
        std::cout << "Saved PNG to: " << filename << std::endl;
    }
}

bool has_png_extension(const std::string &filename) {
    if (filename.length() < 4) return false;
    return filename.substr(filename.length() - 4) == ".png";
}

std::vector<float> read_all_pngs_in_folder(const std::string &folder_path) {
    std::vector<float> all_data;

    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(folder_path.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string filename = ent->d_name;
            if (has_png_extension(filename)) {
                std::string full_path = folder_path + "/" + filename;
                unsigned width, height;
                std::vector<float> img_data = read_png_as_float(full_path, width, height);
                all_data.insert(all_data.end(), img_data.begin(), img_data.end());
                std::cout << "Loaded " << filename << " (" << width << "x" << height << ")" << std::endl;
            }
        }
        closedir(dir);
    } else {
        throw std::runtime_error("Cannot open directory: " + folder_path);
    }

    return all_data;
}


void run_inference(float* input_data, const std::string &model_path, float threshold) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNX_Runtime");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    std::string model = "./models/"+model_path;
    Ort::Session session(env, model.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    auto input_type_info = session.GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_dims = input_tensor_info.GetShape();

    if (input_dims.at(0) == -1) input_dims[0] = 1; // Dynamic batch size

    if (input_dims != std::vector<int64_t>{1, 4, 384, 384})
        throw std::runtime_error("Invalid input shape");

    size_t input_size = 1 * 4 * 384 * 384;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data, input_size, input_dims.data(), input_dims.size());

    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    std::vector<const char*> input_names = {input_name.get()};
    std::vector<const char*> output_names = {output_name.get()};

    std::vector<Ort::Value> output_tensors = session.Run(
        Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1,
        output_names.data(), 1);

    float* results = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    size_t output_size = 1;
    for (auto dim : output_shape) output_size *= dim;

    std::cout << "Output tensor shape: [";
    for (size_t i = 0; i < output_shape.size(); ++i) {
        std::cout << output_shape[i];
        if (i != output_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    size_t width = output_shape.back();
    size_t height = output_shape[output_shape.size() - 2];

    std::string full_output = "./results/"+model_path+"_t_"+std::to_string(threshold)+"_onnx_cpp.png";

    save_grayscale_png_from_float(results, width, height, full_output, threshold);
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " <input_folder> <model_path> [threshold]" << std::endl;
            return 1;
        }

        std::string input_folder = argv[1];
        std::string model_path = std::string(argv[2]);

        float threshold = 0.5f;  // default

        if (argc >= 4) {
            char* end;
            threshold = std::strtof(argv[3], &end);
            if (*end != '\0') {
                std::cerr << "Invalid threshold value: " << argv[3] << std::endl;
                return 1;
            }
        }

        std::cout << "Input folder: " << input_folder << "\n";
        std::cout << "Model path: " << model_path << "\n";
        std::cout << "Threshold: " << threshold << "\n";

        std::vector<float> image_data = read_all_pngs_in_folder(input_folder);

        if (!image_data.empty()) {
            run_inference(image_data.data(), model_path, threshold);
        } else {
            std::cerr << "No image data found in folder: " << input_folder << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

