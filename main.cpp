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

inline __int64_t continuousTimeNs()
{
	timespec now;
	clock_gettime(CLOCK_REALTIME, &now);
	return (__int64_t ) now.tv_sec * 1000000000 + (__int64_t ) now.tv_nsec;
}

using namespace std;

#define GUI
#define NUM_FRAMES 2000

#define THREADS_PER_BLOCK 128
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
	hack<<<1, 1>>>(gpuVelocities, gpuAcceleration, gpuPositions, gpuMasses);
}

void callback () {

}

int main(int argc, char **argv) {

	if (argc != 2) {
		cout << "Usage: " << argv[0] << " <numBodies>" << endl;
		return 1;
	}
	unsigned int numBodies = atoi(argv[1]);
	unsigned int numBlocks = numBodies / THREADS_PER_BLOCK;
	numBodies = numBlocks * THREADS_PER_BLOCK;

	// allocate memory
	float2 *hPositions = new float2[numBodies];
	float2 *hVelocities = new float2[numBodies];
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

	cudaMalloc((void**) &gpuPositions, numBodies * sizeof(float2));
	cudaMalloc((void**) &gpuVelocities, numBodies * sizeof(float2));
	cudaMalloc((void**) &gpuAcceleration, numBodies * sizeof(float2));
	cudaMalloc((void**) &gpuMasses, numBodies * sizeof(float));

	cudaMemcpy(gpuPositions, hPositions, numBodies * sizeof(float2), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuVelocities, hVelocities, numBodies * sizeof(float2), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuMasses, hMasses, numBodies * sizeof(float), cudaMemcpyHostToDevice);

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

    dim3 blockGrid(numBlocks);
    dim3 threadBlock(THREADS_PER_BLOCK);
    int shared_size = THREADS_PER_BLOCK * sizeof(float4);

	// Calculate
	for (unsigned int t = 0; t < NUM_FRAMES; t++) {
		__int64_t computeStart = continuousTimeNs();

		applyForces<<<blockGrid, threadBlock, shared_size>>>(gpuAcceleration, gpuPositions, gpuMasses, numBodies);
		update<<<blockGrid, threadBlock>>>(gpuVelocities, gpuPositions, gpuAcceleration, numBodies);

		cudaDeviceSynchronize();
		int ns = (continuousTimeNs() - computeStart);
		frametimes[t].y = min((ns / 1000000.0) / 100.0, 0.2) - 1.0;
		frametimes[t].x = ((float)t/NUM_FRAMES) * 2.0 - 1.0 ;
		cout << "Frame compute time: " << ns << "ns" << endl;

		// TODO 5: Download the updated positions into the hPositions array for rendering.
		cudaMemcpy(hPositions, gpuPositions, numBodies * sizeof(float2), cudaMemcpyDeviceToHost);

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
		float2 rect = make_float2(-1.0, -1.0);
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

	cudaFree(gpuVelocities);
	cudaFree(gpuMasses);
	cudaFree(gpuPositions);
	cudaFree(gpuAcceleration);

	delete[] hMasses;
	delete[] hPositions;
	delete[] frametimes;
	delete[] particleWithColor;
}

