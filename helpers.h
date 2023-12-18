#include <GL/glew.h>
#include <GL/gl.h>
#include <ostream>
#include <iostream>
#include <sstream>
#include <tuple>
#include <CL/cl.hpp>
#include <fstream>

std::vector<std::tuple<GLenum, std::string, std::string>> loadShaderContent(std::string vertex_path, std::string fragment_path) {
	std::stringstream vertex;
	vertex << std::ifstream(vertex_path).rdbuf();
	std::stringstream fragment;
	fragment << std::ifstream(fragment_path).rdbuf();
	return {
		std::make_tuple(GL_VERTEX_SHADER, vertex.str(), vertex_path),
		std::make_tuple(GL_FRAGMENT_SHADER, fragment.str(), fragment_path),
	};
}

template <typename T, size_t size>
void addToVector(std::vector<T>* vector, const T (&array)[size]) {
   std::copy(std::begin(array), std::end(array), std::back_insert_iterator(*vector));
}

bool loadShaders(GLuint* program, std::vector<std::tuple<GLenum, std::string, std::string>> shaders) {
	for (const auto& s : shaders) {
		GLenum type = std::get<0>(s);
		const std::string& source = std::get<1>(s);

		const GLchar* src = source.c_str();

		GLuint shader = glCreateShader(type);
		glShaderSource(shader, 1, &src, nullptr);
		glCompileShader(shader);

		GLint compiled = 0;
		glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
		if (!compiled) {
			GLint length = 0;
			glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);

			if (length > 1) {
				std::string log(length, '\0');
				glGetShaderInfoLog(shader, length, &length, &log[0]);
				printf("Shader (%s) compile failed:\n%s\n", source.c_str(), log.c_str());
			}
			else {
				printf("Shader (%s) compile failed.\n", source.c_str());
			}

			return false;
		}

		glAttachShader(*program, shader);
	}

	glLinkProgram(*program);

	GLint linked = 0;
	glGetProgramiv(*program, GL_LINK_STATUS, &linked);

	if (!linked) {
		GLint length = 0;
		glGetProgramiv(*program, GL_INFO_LOG_LENGTH, &length);

		if (length > 1) {
			std::string log(length, '\0');
			glGetProgramInfoLog(*program, length, &length, &log[0]);
			printf("Program (%s) link failed:\n%s", std::get<2>(shaders[0]).c_str(), log.c_str());
		}
		else {
			printf("Program (%s) link failed.\n", std::get<2>(shaders[0]).c_str());
		}

		return false;
	}
	return true;
}

void PrintShaderInfoLog(GLint const Shader)
{
	int InfoLogLength = 0;
	int CharsWritten = 0;

	glGetShaderiv(Shader, GL_INFO_LOG_LENGTH, & InfoLogLength);

	if (InfoLogLength > 0)
	{
		GLchar * InfoLog = new GLchar[InfoLogLength];
		glGetShaderInfoLog(Shader, InfoLogLength, & CharsWritten, InfoLog);
		std::cout << "Shader Info Log:" << std::endl << InfoLog << std::endl;
		delete [] InfoLog;
	}
}

void GLAPIENTRY
MessageCallback( GLenum source,
                 GLenum type,
                 GLuint id,
                 GLenum severity,
                 GLsizei length,
                 const GLchar* message,
                 const void* userParam )
{
  fprintf( stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
           ( type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : "" ),
            type, severity, message );
}

const char *getErrorString(cl_int error)
{
switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}

cl_device_id create_device() {
   cl_platform_id platform;
   cl_device_id dev;
   int err;

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
       printf("Couldn't identify a platform: %s\n", getErrorString(err));
      exit(1);
   } 

   // GPU
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
      // CPU
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
   }
   if(err < 0) {
      printf("Couldn't access any devices: %s\n", getErrorString(err));
      exit(1);   
   }

   return dev;
}

cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {
   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   int err;

   /* Read program file and place content into buffer */
   program_handle = fopen(filename, "r");
   if(program_handle == NULL) {
      perror("Couldn't find the program file");
      exit(1);
   }
   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char*)malloc(program_size + 1);
   program_buffer[program_size] = '\0';
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);

   program = clCreateProgramWithSource(ctx, 1, (const char**)&program_buffer, &program_size, &err);
   if(err < 0) {
      printf("Couldn't create the CL program: %s\n", getErrorString(err));
      exit(1);
   }
   free(program_buffer);

   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if(err < 0) {

      /* Find size of log and print to std output */
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
      program_log = (char*) malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }

   return program;
}

void list_devices() {
   std::vector<cl::Platform> platforms;
   cl::Platform::get(&platforms);

   int platform_id = 0;
   int device_id = 0;

   std::cout << "Number of Platforms: " << platforms.size() << std::endl;

   for(std::vector<cl::Platform>::iterator it = platforms.begin(); it != platforms.end(); ++it) {
      cl::Platform platform(*it);

      std::cout << "Platform ID: " << platform_id++ << std::endl;
      std::cout << "Platform Name: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
      std::cout << "Platform Vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;

      std::vector<cl::Device> devices;
      platform.getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);

      for(std::vector<cl::Device>::iterator it2 = devices.begin(); it2 != devices.end(); ++it2) {
      cl::Device device(*it2);

      std::cout << std::endl << "\tDevice " << device_id++ << ": " << std::endl;
      std::cout << "\t\tDevice Name: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
      std::cout << "\t\tDevice Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
      std::cout << "\t\tDevice Version: " << device.getInfo<CL_DEVICE_VERSION>() << std::endl;
      switch (device.getInfo<CL_DEVICE_TYPE>()) {
         case 4:
            std::cout << "\t\tDevice Type: GPU" << std::endl;
            break;
         case 2:
            std::cout << "\t\tDevice Type: CPU" << std::endl;
            break;
         default:
            std::cout << "\t\tDevice Type: unknown" << std::endl;
      }
      std::cout << "\t\tDevice Max Compute Units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
      std::cout << "\t\tDevice Global Memory: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << std::endl;
      std::cout << "\t\tDevice Max Clock Frequency: " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl;
      std::cout << "\t\tDevice Max Memory Allocation: " << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << std::endl;
      std::cout << "\t\tDevice Local Memory: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
      std::cout << "\t\tDevice Available: " << device.getInfo<CL_DEVICE_AVAILABLE>() << std::endl;
      }
      std::cout << std::endl;
   }
}