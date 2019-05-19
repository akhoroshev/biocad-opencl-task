#define __CL_ENABLE_EXCEPTIONS

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else

#include <CL/cl.hpp>

#endif

#include "config.h"
#include "utils.h"
#include <cmath>
#include <queue>

// algorithm without using OpenCL
void defaultAlgorithm(const std::vector<cl_float4> &atomPositions,
                      const std::vector<cl_float> &atomCharges,
                      const std::vector<cl_int> &atomDegree,
                      const std::vector<cl_int> &atomEdges) {

  std::cout << std::endl
            << "-------------------------------------------------" << std::endl;
  std::cout << "Default algorithm without using OpenCL" << std::endl
            << std::endl;
  const size_t atomCount = atomPositions.size();
  std::cout << "Atoms loaded: " << atomCount << std::endl;

  Timer timer;
  timer.start();
  std::vector<float> scale;
  scale.assign(atomCount * atomCount, 1.0);
  timer.stop("Time for preparing buffers");

  timer.start();
  float s[3] = {0, 0, 0.5};
  std::deque<int> q;
  for (int v = 0; v < atomCount; v++) {
    q.clear();
    q.push_back(v);
    scale[atomCount * v + v] = 0.0;
    for (int iter = 0; iter < 3; iter++) {
      int q_size = q.size();
      for (int i = 0; i < q_size; i++) {
        int pop = q.front();
        q.pop_front();
        for (int j = 0; j < atomDegree[pop]; j++) {
          int toVertex = atomEdges[pop * atomCount + j];
          if (scale[v * atomCount + toVertex] < 1.0)
            continue;
          scale[v * atomCount + toVertex] = s[iter];
          q.push_back(toVertex);
        }
      }
    }
  }

  float sumE = 0;

  for (int i = 0; i < atomCount; i++) {
    for (int j = i + 1; j < atomCount; j++) {
      float val =
          C * scale[i * atomCount + j] * atomCharges[i] * atomCharges[j];
      cl_float4 r1 = atomPositions[i];
      cl_float4 r2 = atomPositions[j];
      float r = (r1.s[0] - r2.s[0]) * (r1.s[0] - r2.s[0]) +
                (r1.s[1] - r2.s[1]) * (r1.s[1] - r2.s[1]) +
                (r1.s[2] - r2.s[2]) * (r1.s[2] - r2.s[2]);
      r = sqrt(r);
      sumE += val / r;
    }
  }
  timer.stop("Time for executing");
  std::cout << "Energy: " << sumE << std::endl;
}

void performCalculation(const cl::Device &device,
                        const std::vector<cl_float4> &atomPositions,
                        const std::vector<cl_float> &atomCharges,
                        const std::vector<cl_int> &atomDegree,
                        const std::vector<cl_int> &atomEdges) {

  std::cout << std::endl
            << "-------------------------------------------------" << std::endl;
  std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl
            << std::endl;

  const size_t atomCount = atomPositions.size();
  std::cout << "Atoms loaded: " << atomCount << std::endl;
  Timer timer;

  cl::Context context(device);
  cl::Program program(context, loadProgram(KERNEL));
  cl::CommandQueue queue(context, device);
  // "-cl-std=CL2.0" for
  // https://software.intel.com/ru-ru/articles/opencl-20-non-uniform-work-groups
  program.build({device}, "-cl-std=CL2.0");

  timer.start();

  // Prepare buffers
  cl::Buffer inputEdges =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 atomEdges.size() * sizeof(cl_int),
                 const_cast<cl_int *>(atomEdges.data()));

  cl::Buffer inputEdgesCount =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 atomDegree.size() * sizeof(cl_int),
                 const_cast<cl_int *>(atomDegree.data()));

  cl::Buffer inputQueueBuffer = cl::Buffer(
      context, CL_MEM_READ_WRITE, atomCount * atomCount * sizeof(cl_int));

  cl::Buffer inputPositions =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 atomPositions.size() * sizeof(cl_float4),
                 const_cast<cl_float4 *>(atomPositions.data()));

  cl::Buffer inputCharges =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 atomCharges.size() * sizeof(cl_float),
                 const_cast<cl_float *>(atomCharges.data()));

  std::vector<cl_float> outputPathScaleInit;
  outputPathScaleInit.assign(atomCount * atomCount, 1.0);
  cl::Buffer outputPathScale =
      cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                 outputPathScaleInit.size() * sizeof(cl_float),
                 outputPathScaleInit.data());

  int workGroupsCount = lround(atomCount / GROUP_SIZE + 0.5);
  std::vector<cl_float> outputEnergyHost;
  outputEnergyHost.resize(workGroupsCount * workGroupsCount);
  cl::Buffer outputEnergy =
      cl::Buffer(context, CL_MEM_READ_WRITE,
                 workGroupsCount * workGroupsCount * sizeof(cl_float));

  timer.stop("Time for preparing buffers");

  // Set kernel args
  cl::Kernel kernelBFS(program, "bfs");
  {
    int iArg = 0;
    kernelBFS.setArg(iArg++, inputEdges);
    kernelBFS.setArg(iArg++, inputEdgesCount);
    kernelBFS.setArg(iArg++, inputQueueBuffer);
    kernelBFS.setArg(iArg, outputPathScale);
  }

  cl::Kernel kernelCoulomb(program, "coulomb");
  {
    int iArg = 0;
    kernelCoulomb.setArg(iArg++, inputCharges);
    kernelCoulomb.setArg(iArg++, inputPositions);
    kernelCoulomb.setArg(iArg++, outputPathScale);
    kernelCoulomb.setArg(iArg, outputEnergy);
  }

  // Run kernel
  timer.start();
  queue.enqueueNDRangeKernel(kernelBFS, cl::NullRange, cl::NDRange(atomCount));
  queue.enqueueNDRangeKernel(kernelCoulomb, cl::NullRange,
                             cl::NDRange(atomCount, atomCount),
                             cl::NDRange(GROUP_SIZE, GROUP_SIZE));
  queue.enqueueReadBuffer(outputEnergy, CL_TRUE, 0,
                          workGroupsCount * workGroupsCount * sizeof(cl_float),
                          outputEnergyHost.data());
  queue.finish();

  // Sum energy
  float sumEnergy = 0;
  for (size_t i = 0; i < workGroupsCount * workGroupsCount; i++) {
    sumEnergy += outputEnergyHost[i];
  }
  timer.stop("Time for executing");

  std::cout << "Energy: " << sumEnergy << std::endl;
}

int main() {
  try {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
      std::cout << "OpenCL platform not found\n";
      return -1;
    }

    std::vector<cl::Device> devices;

    for (const auto &platform : platforms) {
      std::vector<cl::Device> devs;
      platform.getDevices(CL_DEVICE_TYPE_ALL, &devs);
      devices.insert(devices.begin(), devs.begin(), devs.end());
    }

    auto atomPositions = loadAtoms(ATOMS);
    auto atomCharges = loadCharges(CHARGES);
    auto [atomEdges, atomDegree] =
        convertMatrix(loadBounds(BOUNDS, atomPositions.size()));

    for (const auto &device : devices) {
      performCalculation(device, atomPositions, atomCharges, atomDegree,
                         atomEdges);
    }
    defaultAlgorithm(atomPositions, atomCharges, atomDegree,
                       atomEdges);

  } catch (const cl::Error &err) {
    std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")"
              << std::endl;
  }

  return 0;
}