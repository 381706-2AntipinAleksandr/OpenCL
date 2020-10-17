#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

size_t GetSourceFromFile(const std::string& _path, char** source_code) {
    size_t number = _path.find_last_of("/\\");
    std::string path = _path.substr(0, number);
    std::ifstream file(path + "/kernels/kernel.cl", std::ios_base::in | std::ios_base::binary);
    if (!file.is_open()) {
        std::cerr << "ERROR: opening resource file /kernels/kernel.cl is failed!" << std::endl;
    }
    std::stringstream source;
    source << file.rdbuf();
    file.close();
    std::string str = source.str();
    (*source_code) = new char[str.size()];
    for (size_t i = 0; i < str.size(); ++i)
        (*source_code)[i] = str[i];
    return str.size();
}

int main(int argc, char** argv) {

    // Platform:
    cl_uint platformCount = 0;
    if (clGetPlatformIDs(0, nullptr, &platformCount)) {
        std::cerr << "ERROR: GetPlatformIds return error in inicialisation" << std::endl;
    }
    cl_platform_id platform;
    if (platformCount) {
        cl_platform_id* platforms = new cl_platform_id[platformCount];
        if (clGetPlatformIDs(platformCount, platforms, nullptr)) {
            std::cerr << "ERROR: GetPlatformIds return error" << std::endl;
        }
        platform = platforms[0];
        delete[] platforms;
    } else {
        std::cerr << "ERROR: No OpenCL Platform" << std::endl;
        return -1;
    }
    char platformName[128];
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION,  128, platformName, nullptr);
    std::cout << platformName << std::endl;

    // Context
    cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform), 0 };
    cl_context context = clCreateContextFromType((properties) ? properties : nullptr,
        CL_DEVICE_TYPE_GPU, nullptr, nullptr, nullptr);
    size_t device_count;
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, nullptr, &device_count);
    
    cl_device_id device;
    if (device_count) {
        cl_device_id* devices = new cl_device_id[device_count];
        clGetContextInfo(context, CL_CONTEXT_DEVICES, device_count, devices, nullptr);
        // clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count, device, nullptr);
        device = devices[0];
        delete[] devices;
    } else {
        std::cerr << "ERROR: No devices" << std::endl;
        return -1;
    }
    std::cout << device << std::endl;
    cl_uint info;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &info, nullptr);
    std::cout << info << std::endl;

    // Command Queue
    int error_code;
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &error_code);
    if (error_code) {
        std::cerr << "ERROR: OpenCL error code in creating command queue: " << error_code << std::endl;
    }

    // Kernel and Program
    char* kernel_source{};
    std::shared_ptr<char> ptr(kernel_source);
    size_t kernel_len = GetSourceFromFile(argv[0], &kernel_source);
    const char* ks = kernel_source;
    cl_program program = clCreateProgramWithSource(context, 1, &ks, &kernel_len, &error_code);
    if (error_code) {
        std::cerr << "ERROR: OpenCL error code in creating program: " << error_code << std::endl;
    }
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    cl_kernel _kernel = clCreateKernel(program, "square", &error_code);
    if (error_code) {
        std::cerr << "ERROR: OpenCL error code in creating kernel: " << error_code << std::endl;
    }

    // input and output values
    const size_t size = 100;
    int input[size];
    int output[size];
    for (int i = 0; i < size; ++i) {
        input[i] = size - i - 1;
    }
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * size, nullptr, &error_code);
    if (error_code) {
        std::cerr << "ERROR: OpenCL error code in creating input buffer: " << error_code << std::endl;
    }
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size, nullptr, &error_code);
    if (error_code) {
        std::cerr << "ERROR: OpenCL error code in creating output buffer: " << error_code << std::endl;
    }
    clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, sizeof(int) * size, input, 0, nullptr, nullptr);

    // Starting kernel
    clSetKernelArg(_kernel, 0, sizeof(cl_mem), &input_buffer);
    clSetKernelArg(_kernel, 1, sizeof(cl_mem), &output_buffer);
    clSetKernelArg(_kernel, 2, sizeof(unsigned int), &size);
    size_t group;
    clGetKernelWorkGroupInfo(_kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, nullptr);
    clEnqueueNDRangeKernel(queue, _kernel, 1, nullptr, &size, &group, 0, nullptr, nullptr);
    clFinish(queue);

    // Coping from kernel
    clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(int) * size, output, 0, nullptr, nullptr);

    // Cleaning memory
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseProgram(program);
    clReleaseKernel(_kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    for (size_t i = 0; i < size; ++i)
        std::cout << output[i] << ", ";
    std::cout << std::endl;

    return 0;
}