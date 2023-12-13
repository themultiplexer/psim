#include <fstream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <math.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/freeglut_std.h>
#include <sys/time.h>
#include <CL/cl.h>

#define PROGRAM_FILE "../kernels.cl"
#define KERNEL_FUNC "applyForces"

/* Find a GPU or CPU associated with the first available platform 

The `platform` structure identifies the first platform identified by the 
OpenCL runtime. A platform identifies a vendor's installation, so a system 
may have an NVIDIA platform and an AMD platform. 

The `device` structure corresponds to the first accessible device 
associated with the platform. Because the second parameter is 
`CL_DEVICE_TYPE_GPU`, this device must be a GPU.
*/
cl_device_id create_device() {

   cl_platform_id platform;
   cl_device_id dev;
   int err;

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
      perror("Couldn't identify a platform");
      exit(1);
   } 

   // GPU
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
      // CPU
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
   }
   if(err < 0) {
      perror("Couldn't access any devices");
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
      perror("Couldn't create the program");
      exit(1);
   }
   free(program_buffer);

   /* Build program 

   The fourth parameter accepts options that configure the compilation. 
   These are similar to the flags used by gcc. For example, you can 
   define a macro with the option -DMACRO=VALUE and turn off optimization 
   with -cl-opt-disable.
   */
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

struct float2 {
  float x, y;
};
struct float4 {
  float x, y, z, w;
};

struct GLParticle {
	GLfloat x;
	GLfloat y;
	GLfloat r;
	GLfloat g;
	GLfloat b;
	GLfloat a;
	GLfloat size;
};

inline __int64_t continuousTimeNs()
{
	timespec now;
	clock_gettime(CLOCK_REALTIME, &now);
	return (__int64_t ) now.tv_sec * 1000000000 + (__int64_t ) now.tv_nsec;
}

using namespace std;

#define GUI
#define NUM_FRAMES 2000

#define THREADS_PER_BLOCK 100
#define EPS_2 0.00001f
#define GRAVITY 0.00000001f

float randF(const float min = 0.0f, const float max = 1.0f) {
	int randI = rand();
	float randF = (float) randI / (float) RAND_MAX;
	float result = min + randF * (max - min);

	return result;
}

std::string readFile(const char *filePath) {
    std::string content;
    std::ifstream fileStream(filePath, std::ios::in);

    if(!fileStream.is_open()) {
        std::cerr << "Could not read file " << filePath << ". File does not exist." << std::endl;
        return "";
    }

    std::string line = "";
    while(!fileStream.eof()) {
        std::getline(fileStream, line);
        content.append(line + "\n");
    }

    fileStream.close();
    return content;
}

bool overlay = true;
bool masses = false;
bool smear = false;

void keyboard( unsigned char key, int x, int y )
{
    if(key == 'd'){
        overlay = !overlay;
    } else if (key == 'm') {
		masses = !masses;
	} else if (key == 's') {
		smear = !smear;
	}
}


float2 *gpuPositions;
float2 *gpuVelocities;
float2 *gpuAcceleration;
float *gpuMasses;

void mouse(int button, int state, int x, int y) {
	printf("%d: %d  %dx%d\n", button, state,x, y);
	//hack<<<1, 1>>>(gpuVelocities, gpuAcceleration, gpuPositions, gpuMasses);
}

void callback () {

}

int main(int argc, char **argv) {
	/*
	if (argc != 2) {
		cout << "Usage: " << argv[0] << " <numBodies>" << endl;
		return 1;
	}
	unsigned int numBodies = atoi(argv[1]);*/

	size_t numBodies = 10000;
	size_t numBlocks = numBodies / THREADS_PER_BLOCK;
	printf("CL_DEVICE_MAX_WORK_ITEM_SIZES:%d \n", CL_DEVICE_MAX_WORK_ITEM_SIZES );
	printf("MAX_WORK_GROUP_SIZE:%d total:%d blocks:%d\n", CL_DEVICE_MAX_WORK_GROUP_SIZE, numBodies, numBlocks);

	// allocate memory
	float2 *hPositions = new float2[numBodies];
	float2 *hPositions2 = new float2[numBodies];
	float2 *hVelocities = new float2[numBodies];
	float2 *hAccelerations = new float2[numBodies];
	float *hMasses = new float[numBodies];
	GLParticle *particleWithColor = new GLParticle[numBodies];
	float2 *frametimes = new float2[NUM_FRAMES];

	// Initialize Positions and speed
	for (unsigned int i = 0; i < numBodies; i++) {
		hPositions[i].x = randF(-1.0, 1.0);
		hPositions[i].y = randF(-1.0, 1.0);
		hVelocities[i].x = hPositions[i].y * 0.007f + randF(0.001f, -0.001f);
		hVelocities[i].y = -hPositions[i].x * 0.007f + randF(0.001f, -0.001f);
		hMasses[i] = randF(0.0f, 1.0f) * 10000.0f / (float) numBodies;
		particleWithColor[i].r = ((float)hPositions[i].x + 1.0f) / 2.f;
		particleWithColor[i].g = 1.0f - particleWithColor[i].r;
		particleWithColor[i].b = ((float)hPositions[i].y + 1.0f) / 2.f;
	}

	cl_int err;
	cl_device_id device = create_device();
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	if(err < 0) {
		perror("Couldn't create a context");
		exit(1);   
	}

   /* Build program */
   cl_program program = build_program(context, device, PROGRAM_FILE);

   cl_mem position_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, numBodies * sizeof(float2), hPositions, &err);
   cl_mem velocity_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, numBodies * sizeof(float2), hVelocities, &err);
   cl_mem acceleration_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, numBodies * sizeof(float2), hAccelerations, &err);
   cl_mem mass_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, numBodies * sizeof(float), hMasses, &err);
   if(err < 0) {
      perror("Couldn't create a buffer");
      exit(1);   
   };

   /* Create a command queue 
   Does not support profiling or out-of-order-execution
   */
   cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
   if(err < 0) {
      perror("Couldn't create a command queue");
      exit(1);   
   };

   cl_kernel applykernel = clCreateKernel(program, "applyForces", &err);
   if(err < 0) {
      perror("Couldn't create a kernel");
      exit(1);
   };

   /* Create kernel arguments */
   err = clSetKernelArg(applykernel, 0, sizeof(cl_mem), &acceleration_buffer);
   err |= clSetKernelArg(applykernel, 1, sizeof(cl_mem), &position_buffer);
   err |= clSetKernelArg(applykernel, 2, THREADS_PER_BLOCK * sizeof(float4), NULL); // Allocate Shared Memory
   err |= clSetKernelArg(applykernel, 3, sizeof(cl_mem), &mass_buffer);
   if(err < 0) {
      perror("Couldn't create a kernel argument");
      exit(1);
   }

  cl_kernel updatekernel = clCreateKernel(program, "update", &err);
   if(err < 0) {
      perror("Couldn't create a kernel");
      exit(1);
   };

   /* Create kernel arguments */
   err = clSetKernelArg(updatekernel, 0, sizeof(cl_mem), &velocity_buffer);
   err |= clSetKernelArg(updatekernel, 1, sizeof(cl_mem), &position_buffer);
   err |= clSetKernelArg(updatekernel, 2, sizeof(cl_mem), &acceleration_buffer);
   if(err < 0) {
      perror("Couldn't create a kernel argument");
      exit(1);
   }

	// Free host memory not needed again
	delete[] hVelocities;
	
	// Initialize OpenGL rendering
#ifdef GUI
	glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE );
    glutInitWindowSize( 2000, 2000 );
    glutCreateWindow("Massively Parallel Computing - NBody Simulation");
	glutDisplayFunc(callback);
    glewInit();
	glEnable(GL_PROGRAM_POINT_SIZE);

	GLuint shaderprogram;
	std::string vertexSource = readFile("../shader.glsl");
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	const GLchar *source = (const GLchar *)vertexSource.c_str();
	glShaderSource(vertexShader, 1, &source, 0);
	glCompileShader(vertexShader);
	shaderprogram = glCreateProgram();
    glAttachShader(shaderprogram, vertexShader);
    glLinkProgram(shaderprogram);
	glUseProgram(shaderprogram);

	GLuint vb;
	glGenBuffers(1, &vb);
	glBindBuffer(GL_ARRAY_BUFFER, vb);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLParticle) * numBodies, particleWithColor, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	GLuint va;
	glGenVertexArrays(1, &va);
	glBindVertexArray(va);
	glBindBuffer(GL_ARRAY_BUFFER, vb);

	glEnableVertexAttribArray(glGetAttribLocation(shaderprogram, "position"));
	glVertexAttribPointer(glGetAttribLocation(shaderprogram, "position"), 2, GL_FLOAT, GL_FALSE, sizeof(GLParticle), (void*)offsetof( GLParticle, x ));
	glEnableVertexAttribArray(glGetAttribLocation(shaderprogram, "color"));
	glVertexAttribPointer(glGetAttribLocation(shaderprogram, "color"), 4, GL_FLOAT, GL_FALSE, sizeof(GLParticle), (void*)offsetof( GLParticle, r ));
	glEnableVertexAttribArray(glGetAttribLocation(shaderprogram, "pointsize"));
	glVertexAttribPointer(glGetAttribLocation(shaderprogram, "pointsize"), 1, GL_FLOAT, GL_FALSE, sizeof(GLParticle), (void*)offsetof( GLParticle, size ));

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	GLuint vbp;
	glGenBuffers(1, &vbp);

	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
#endif

    int shared_size = THREADS_PER_BLOCK * sizeof(float4);
    size_t tpb = THREADS_PER_BLOCK;

	// Calculate
	for (unsigned int t = 0; t < NUM_FRAMES; t++) {
		__int64_t computeStart = continuousTimeNs();

		//applyForces<<<blockGrid, threadBlock, shared_size>>>(gpuAcceleration, gpuPositions, gpuMasses, numBodies);
		//update<<<blockGrid, threadBlock>>>(gpuVelocities, gpuPositions, gpuAcceleration, numBodies);

		err = clEnqueueNDRangeKernel(queue, applykernel, 1, NULL, &numBodies, &tpb, 0, NULL, NULL); 
		if(err < 0) {
			printf("Couldn't enqueue the apply kernel: %s\n", getErrorString(err));
			exit(1);
		}
		err = clEnqueueNDRangeKernel(queue, updatekernel, 1, NULL, &numBodies, &tpb, 0, NULL, NULL); 
		if(err < 0) {
			printf("Couldn't enqueue the update kernel: %s\n", getErrorString(err));
			exit(1);
		}

		int ns = (continuousTimeNs() - computeStart);
		frametimes[t].y = min((ns / 1000000.0) / 100.0, 0.2) - 1.0;
		frametimes[t].x = ((float)t/NUM_FRAMES) * 2.0 - 1.0 ;
		cout << "Frame compute time: " << ns << "ns" << endl;

		err = clEnqueueReadBuffer(queue, position_buffer, CL_TRUE, 0, numBodies * sizeof(float2), hPositions, 0, NULL, NULL);
		if(err < 0) {
			printf("Couldn't read the buffer: %s\n", getErrorString(err));
			exit(1);
		}

		// TODO 5: Download the updated positions into the hPositions array for rendering.
		//cudaMemcpy(hPositions, gpuPositions, numBodies * sizeof(float2), cudaMemcpyDeviceToHost);

#ifdef GUI

	for (unsigned int i = 0; i < numBodies; i++) {
		particleWithColor[i].x = hPositions[i].x;
		particleWithColor[i].y = hPositions[i].y;
		if (masses) {
			particleWithColor[i].size = pow(hMasses[i], 3) * 10.0;
			//particleWithColor[i].a = hMasses[i];
		}else {
			particleWithColor[i].size = 10.0;
			particleWithColor[i].a = 1.0;
		}
	}

	glBindBuffer(GL_ARRAY_BUFFER, vb);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLParticle) * numBodies, particleWithColor, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	if(!smear){
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	} else {
		glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}
	
	glUseProgram(shaderprogram);
	glBindVertexArray(va);
	glDrawArrays(GL_POINTS, 0, numBodies);
	glBindVertexArray(0);
	glUseProgram(0);

	if(overlay){
		// Draw Info rectangle
		float2 rect = float2{-1.0, -1.0};
		glPushMatrix();
		glTranslatef(rect.x, rect.y, 0.0f);
		glBegin(GL_QUADS);
		glColor4f(1.0, 1.0, 1.0, 0.75);
		glVertex2f(0, 0);
		glVertex2f(0, 0.1);
		glVertex2f(2.0, 0.1);
		glVertex2f(2-0, 0);       
		glEnd();
		glPopMatrix();

		// Draw frametime string
		glColor4f(0.0, 0.0, 0.0, 1.0);
		glWindowPos2i(10, 50);
		glutBitmapString(GLUT_BITMAP_HELVETICA_18, (unsigned const char*)( std::to_string((ns / 1000000.0)).c_str() ) );

		// Draw graph
		glBindBuffer(GL_ARRAY_BUFFER, vbp);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * t, frametimes, GL_STATIC_DRAW);
		glColor4f(0.0, 0.0, 0.0, 1.0);
		glVertexPointer(2, GL_FLOAT, sizeof(float2), 0);
		glEnableClientState(GL_VERTEX_ARRAY);
		glDrawArrays(GL_LINE_STRIP, 0, t);
		glDisableClientState(GL_VERTEX_ARRAY);
		glLineWidth(2.0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	glutMainLoopEvent();
    glutSwapBuffers();
#endif
	}

#ifdef GUI
	cout << "Done." << endl;
#endif

#ifdef GUI
	glDeleteBuffers(1, &vb);
	glDeleteBuffers(1, &va);
	glDeleteBuffers(1, &vbp);
#endif

   /* Deallocate resources */
   clReleaseKernel(updatekernel);
   clReleaseKernel(applykernel);
   clReleaseMemObject(mass_buffer);
   clReleaseMemObject(position_buffer);
   clReleaseMemObject(velocity_buffer);
   clReleaseMemObject(acceleration_buffer);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);

	delete[] hMasses;
	delete[] hPositions;
	delete[] frametimes;
	delete[] particleWithColor;
}

