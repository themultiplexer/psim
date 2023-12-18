#include "helpers.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/freeglut_std.h>
#include <chrono>
#include <CL/cl.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc.hpp>

#define PROGRAM_FILE "../kernels.cl"
#define MIN_MASS 0.0
#define MAX_MASS 1.0
#define KERNEL_FUNC "applyForces"
#define WINDOW_WIDTH 3840
#define WINDOW_HEIGHT 2160
#define CLI_INPUT false


std::vector<float> generateSubdividedPlaneVertices(int rows, int cols) {
    std::vector<float> vertices;

    const float xstep = 1.0f / static_cast<float>(cols);
    const float ystep = 1.0f / static_cast<float>(rows);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float x1 = j * xstep * 2.0 - 1.0;
            float x2 = (j+1) * xstep * 2.0 - 1.0;
            float y1 = i * ystep * 2.0 - 1.0;
            float y2 = (i+1) * ystep * 2.0 - 1.0;
            float z = 0.0f;
            // Create sub-plane (two triangles) with barycentric coordinates 
            addToVector(&vertices, {x1,y1,z});
            addToVector(&vertices, {0.0f,0.0f,1.0f});
            addToVector(&vertices, {x1,y2,z});
            addToVector(&vertices, {0.0f,1.0f, 0.0f});
            addToVector(&vertices, {x2,y1,z});
            addToVector(&vertices, {1.0f,0.0f, 0.0f});
            addToVector(&vertices, {x1,y2,z});
            addToVector(&vertices, {0.0f,0.0f,1.0f});
            addToVector(&vertices, {x2,y1,z});
            addToVector(&vertices, {0.0f,1.0f, 0.0f});
            addToVector(&vertices, {x2,y2,z});
            addToVector(&vertices, {1.0f,0.0f, 0.0f});
        }
    }

    return vertices;
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
	GLfloat mass;
};

using namespace std;

#define GUI
#define NUM_FRAMES 2000

#define THREADS_PER_BLOCK 500
#define EPS_2 0.00001f
#define GRAVITY 0.00000001f

cv::UMat u3;
GLuint shaderprogram, gravprogram, textureprogram;

bool overlay = false;
bool masses = true;
bool smear = false;
float currentZoom = 0.8f;

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

void keyboard( unsigned char key, int x, int y )
{
    if(key == 'd'){
        overlay = !overlay;
    } else if (key == 'm') {
		masses = !masses;
	} else if (key == 's') {
		smear = !smear;
	} else if (key == 27) {
		glutLeaveMainLoop();
	}
   
}

int ox, oy;
GLfloat offset[2] = {0.0, 0.0};

void mouse(int button, int state, int x, int y) {
	if ((button == 3) || (button == 4)) // It's a wheel event
	{
		if (state == GLUT_UP) return;

		if (button == 3) {
			currentZoom *= 1.2;
         offset[0] -= ((((float)(x) / WINDOW_WIDTH) - 0.5) * 3.55) / currentZoom;
         offset[1] += ((((float)(y) / WINDOW_HEIGHT) - 0.5) * 2.0) / currentZoom;
		} else {
			currentZoom /= 1.2;
         offset[0] = 0.0;
         offset[1] = 0.0;
		}

      printf("%d, %d, %f\n", x, y, currentZoom);
		glUseProgram(shaderprogram);
		glUniform1f(glGetUniformLocation(shaderprogram, "zoom"), currentZoom);
      glUniform2fv(glGetUniformLocation(shaderprogram, "offset"), 1, offset);
		glUseProgram(0);
      u3 = cv::UMat(cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT), CV_8UC4);
	} else {  // normal button event
      if (state == GLUT_DOWN) {
         // Start dragging
         ox = x;
         oy = y;
      } else {
         // Stop dragging
         offset[0] += (float)(x - ox) / WINDOW_WIDTH;
         offset[1] -= (float)(y - oy) / WINDOW_HEIGHT;
      }
		printf("Button %s At %d %d\n", (state == GLUT_DOWN) ? "Down" : "Up", x, y);
	}
	
	
	// TODO Add option again to "spawn" particle with mass
	//hack<<<1, 1>>>(gpuVelocities, gpuAcceleration, gpuPositions, gpuMasses);
}

void mouseMove(int mx, int my) {
   float dx = (float)(mx - ox) / WINDOW_WIDTH;
   float dy = (float)(my - oy) / WINDOW_HEIGHT;
   
   printf("Move At %f %f\n", dx, dy);
   glUseProgram(shaderprogram);
   glUniform2fv(glGetUniformLocation(shaderprogram, "offset"), 1, new GLfloat[2]{offset[0] + dx, offset[1] - dy});
   glUseProgram(0);
}

void callback () {

}

void on_resize(int w, int h)
{
   glViewport(0, 0, w, h);
}

int main(int argc, char **argv) {
   list_devices();

	cout << "Press s for smear, m for mass-mode, scroll = zoom, d for debug overlay" << endl;
   size_t numBodies = 10000;

	if (CLI_INPUT) {
      if (argc != 2) {
         cout << "Usage: " << argv[0] << " <numBodies>" << endl;
         cout << "---Note: <numBodies> must be dividable by 100" << endl;
         return 1;
	   }
   }
   
   if (argc == 2) {
      size_t numBodies = atoi(argv[1]);
   }
   
	size_t numBlocks = numBodies / THREADS_PER_BLOCK;
	printf("CL_DEVICE_MAX_WORK_ITEM_SIZES:%d \n", CL_DEVICE_MAX_WORK_ITEM_SIZES );
	printf("MAX_WORK_GROUP_SIZE:%d total:%zd blocks:%zd\n", CL_DEVICE_MAX_WORK_GROUP_SIZE, numBodies, numBlocks);

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
		hMasses[i] = randF(MIN_MASS, MAX_MASS); // Change limits for interesting stuff to happen :)
		particleWithColor[i].r = ((float)hPositions[i].x + 1.0f) / 2.f;
		particleWithColor[i].g = 1.0f - particleWithColor[i].r;
		particleWithColor[i].b = ((float)hPositions[i].y + 1.0f) / 2.f;
		particleWithColor[i].mass = hMasses[i];
	}

	cl_int err;
	cl_device_id device = create_device();
   cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	if(err < 0) {
      printf("Couldn't create a context: %s\n", getErrorString(err));
		exit(1);   
	}

   /* Build program */
   cl_program program = build_program(context, device, PROGRAM_FILE);

   cl_mem position_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, numBodies * sizeof(float2), hPositions, &err);
   cl_mem velocity_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, numBodies * sizeof(float2), hVelocities, &err);
   cl_mem acceleration_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, numBodies * sizeof(float2), hAccelerations, &err);
   cl_mem mass_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, numBodies * sizeof(float), hMasses, &err);
   if(err < 0) {
      printf("Couldn't create a buffer: %s\n", getErrorString(err));
      exit(1);   
   };

   cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
   if(err < 0) {
      printf("Couldn't create a command queue: %s\n", getErrorString(err));
      exit(1);   
   };

   cl_kernel applykernel = clCreateKernel(program, "applyForces", &err);
   if(err < 0) {
      printf("Couldn't create a kernel: %s\n", getErrorString(err));
      exit(1);
   };

   /* Create kernel arguments */
   err = clSetKernelArg(applykernel, 0, sizeof(cl_mem), &acceleration_buffer);
   err |= clSetKernelArg(applykernel, 1, sizeof(cl_mem), &position_buffer);
   err |= clSetKernelArg(applykernel, 2, THREADS_PER_BLOCK * sizeof(float4), NULL); // Allocate Shared Memory
   err |= clSetKernelArg(applykernel, 3, sizeof(cl_mem), &mass_buffer);
   if(err < 0) {
      printf("Couldn't create a kernel argument: %s\n", getErrorString(err));
      exit(1);
   }

   cl_kernel updatekernel = clCreateKernel(program, "update", &err);
   if(err < 0) {
      printf("Couldn't create a kernel: %s\n", getErrorString(err));
      exit(1);
   };

   /* Create kernel arguments */
   err = clSetKernelArg(updatekernel, 0, sizeof(cl_mem), &velocity_buffer);
   err |= clSetKernelArg(updatekernel, 1, sizeof(cl_mem), &position_buffer);
   err |= clSetKernelArg(updatekernel, 2, sizeof(cl_mem), &acceleration_buffer);
   if(err < 0) {
      printf("Couldn't create a kernel argument: %s\n", getErrorString(err));
      exit(1);
   }

	// Free host memory not needed again
	delete[] hVelocities;
	
	// Initialize OpenGL rendering
	glutInit( &argc, argv );
   glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE );
   
   //glutInitWindowSize( WINDOW_WIDTH, WINDOW_HEIGHT );
   glutInitWindowPosition(100,100);
   glutCreateWindow("Massively Parallel Computing - NBody Simulation");
   glutFullScreen();
	glutDisplayFunc(callback);
   glutReshapeFunc(on_resize);
   
   glewInit();
   glEnable(GL_DEBUG_OUTPUT);
   glDebugMessageCallback(MessageCallback, 0);

   if (cv::ocl::haveOpenCL()){
		cv::ogl::ocl::initializeContextFromGL();
      cv::ocl::setUseOpenCL(true);
	}else {
      printf("Bruh No OpenCV<->OpenGL interop :/");
      exit(123);
   }
	
   printf("Creating particle shader \n");
	shaderprogram = glCreateProgram();
	if (!loadShaders(&shaderprogram, loadShaderContent("../particles.vert.glsl", "../particles.frag.glsl"))) {
		return -1;
	}
	glUseProgram(shaderprogram);
	GLfloat uResolution[2] = { (float)WINDOW_WIDTH, (float)WINDOW_HEIGHT };
   glUniform2fv(glGetUniformLocation(shaderprogram, "iResolution"), 1, uResolution);
   glUniform1f(glGetUniformLocation(shaderprogram, "zoom"), currentZoom);
   glUseProgram(0);

   printf("Creating gravitation field shader \n");
	gravprogram = glCreateProgram();
	if (!loadShaders(&gravprogram, loadShaderContent("../gravitational.vert.glsl", "../gravitational.frag.glsl"))) {
		return -1;
	}

   printf("Creating program \n");
	textureprogram = glCreateProgram();
	if (!loadShaders(&textureprogram, loadShaderContent("../texture.vert.glsl", "../texture.frag.glsl"))) {
		return -1;
	}

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
	glEnableVertexAttribArray(glGetAttribLocation(shaderprogram, "mass"));
	glVertexAttribPointer(glGetAttribLocation(shaderprogram, "mass"), 1, GL_FLOAT, GL_FALSE, sizeof(GLParticle), (void*)offsetof( GLParticle, mass ));

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

   // Set up vertex data for the subdivided plane
   const int rows = 90;
   const int cols = 160;
   std::vector<float> planeVertices = generateSubdividedPlaneVertices(rows, cols);
   GLuint planeVAO, planeVBO;
   glGenVertexArrays(1, &planeVAO);
   glGenBuffers(1, &planeVBO);
   glBindVertexArray(planeVAO);
   glBindBuffer(GL_ARRAY_BUFFER, planeVBO);
   glBufferData(GL_ARRAY_BUFFER, planeVertices.size() * sizeof(float), planeVertices.data(), GL_STATIC_DRAW);
   glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
   glEnableVertexAttribArray(0);
   glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (GLvoid*)(3 * sizeof(float)));
   glEnableVertexAttribArray(1);
   glBindBuffer(GL_ARRAY_BUFFER, 0);
   glBindVertexArray(0);

   float x = 1.0;
   GLfloat quads[12] = {-x,-x, -x,x, x,x, -x,-x, x,-x, x,x };
   GLuint textureVAO, textureVBO;
   glGenVertexArrays(1, &textureVAO);
   glGenBuffers(1, &textureVBO);
   glBindVertexArray(textureVAO);
   glBindBuffer(GL_ARRAY_BUFFER, textureVBO);
   glBufferData(GL_ARRAY_BUFFER, sizeof(quads), quads, GL_STATIC_DRAW);
   glUseProgram(textureprogram);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
   glEnableVertexAttribArray(0);
   glBindBuffer(GL_ARRAY_BUFFER, 0);
   glBindVertexArray(0);
   glUseProgram(0);

	GLuint vbp;
	glGenBuffers(1, &vbp);

   cv::ogl::Texture2D texture;
   GLuint fbo, fbo_texture;

   glActiveTexture(GL_TEXTURE0);
   glGenTextures(1, &fbo_texture);
   glBindTexture(GL_TEXTURE_2D, fbo_texture);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
   texture = cv::ogl::Texture2D(cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT), cv::ogl::Texture2D::Format::RGBA, fbo_texture, false);
   glBindTexture(GL_TEXTURE_2D, 0);

   /* Framebuffer to link everything together */
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbo_texture, 0);

   GLenum status;
   if ((status = glCheckFramebufferStatus(GL_FRAMEBUFFER)) != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "glCheckFramebufferStatus: error %p",
         glewGetErrorString(status));
      return 0;
   }
   glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glEnable(GL_POINT_SMOOTH);
   glEnable(GL_PROGRAM_POINT_SIZE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
   glutMotionFunc(mouseMove);

   size_t tpb = THREADS_PER_BLOCK;

   u3 = cv::UMat(cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT), CV_8UC4);

	// Calculate
	for (unsigned int t = 0; t < NUM_FRAMES; t++) {
		auto computeStart = std::chrono::high_resolution_clock::now();

		err = clEnqueueNDRangeKernel(queue, applykernel, 1, NULL, &numBodies, &tpb, 0, NULL, NULL); 
		err = clEnqueueNDRangeKernel(queue, updatekernel, 1, NULL, &numBodies, &tpb, 0, NULL, NULL); 

      // OpenCV postprocessing on texture
      cv::UMat u1, u2;
		cv::ogl::convertFromGLTexture2D(texture, u1);
      if(!smear){
         cv::blur(u1, u2, cv::Size(61, 61));
         //cv::GaussianBlur(u1, u2, cv::Size(61, 61), 0.0);
         cv::multiply(u2, 3.0, u2);
         cv::add(u1, u2, u2);
         cv::ogl::convertToGLTexture2D(u2, texture);
      } else {
         cv::subtract(u3, u1, u3);
         cv::add(u1, u3, u2);
         cv::ogl::convertToGLTexture2D(u2, texture);
			cv::add(u1, u3, u3);
         cv::multiply(u3, 0.98, u3);
		}

      glUseProgram(gravprogram);
		glBindVertexArray(planeVAO);
		glDrawArrays(GL_TRIANGLES, 0, rows * cols * 6);
		glBindVertexArray(0);
		glUseProgram(0);

      glUseProgram(textureprogram);
		glBindVertexArray(textureVAO);
      texture.bind();
		glDrawArrays(GL_TRIANGLES, 0, 6);
		glBindVertexArray(0);
		glUseProgram(0);


		clFinish(queue);
		int ns = (std::chrono::high_resolution_clock::now() - computeStart).count();
		frametimes[t].y = min((ns / 1000000.0) / 100.0, 0.2) - 1.0;
		frametimes[t].x = ((float)t/NUM_FRAMES) * 2.0 - 1.0 ;
		//cout << "Frame compute time: " << ns << "ns" << endl;

		err = clEnqueueReadBuffer(queue, position_buffer, CL_TRUE, 0, numBodies * sizeof(float2), hPositions, 0, NULL, NULL);
		if(err < 0) {
			printf("Couldn't read the buffer: %s\n", getErrorString(err));
			exit(1);
		}

		for (unsigned int i = 0; i < numBodies; i++) {
			particleWithColor[i].x = hPositions[i].x;
			particleWithColor[i].y = hPositions[i].y;
			if (masses) {
				particleWithColor[i].size = ((hMasses[i] - MIN_MASS) / (MAX_MASS - MIN_MASS)) * 20.0;
				particleWithColor[i].a = 1.0;
			}else {
				particleWithColor[i].size = 10.0;
				particleWithColor[i].a = 1.0;
			}
		}

		glBindBuffer(GL_ARRAY_BUFFER, vb);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLParticle) * numBodies, particleWithColor, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

      // Write to Framebuffer Texture
      glBindFramebuffer(GL_FRAMEBUFFER, fbo);
      glClearColor((GLclampf) 0.0f,(GLclampf) 0.0f,(GLclampf) 0.0f,(GLclampf) 0.0f);
      glClear(GL_COLOR_BUFFER_BIT);
		glUseProgram(shaderprogram);
		glBindVertexArray(va);
		glDrawArrays(GL_POINTS, 0, numBodies);
		glBindVertexArray(0);
		glUseProgram(0);
      glBindFramebuffer(GL_FRAMEBUFFER, 0);



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
			glWindowPos2i(10, 20);
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
	}

	cout << "Done." << endl;

	glDeleteBuffers(1, &vb);
	glDeleteBuffers(1, &va);
	glDeleteBuffers(1, &vbp);

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

