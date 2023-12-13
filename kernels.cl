inline __device__ float2 operator+(const float2 op1, const float2 op2) {
	return make_float2(op1.x + op2.x, op1.y + op2.y);
}

inline __device__ float2 operator-(const float2 op1, const float2 op2) {
	return make_float2(op1.x - op2.x, op1.y - op2.y);
}

inline __device__ float2 operator*(const float2 op1, const float op2) {
	return make_float2(op1.x * op2, op1.y * op2);
}

inline __device__ float2 operator/(const float2 op1, const float op2) {
	return make_float2(op1.x / op2, op1.y / op2);
}

inline __device__ void operator+=(float2 &a, const float2 b) {
	a.x += b.x;
	a.y += b.y;
}

struct GLParticle {
	GLfloat x;
	GLfloat y;
	GLfloat r;
	GLfloat g;
	GLfloat b;
	GLfloat a;
	GLfloat size;
};

__global__ void applyForces(float2 *accelerations, float2 *positions, float *masses, const int n) {
	extern __shared__ float4 sharedposmass[];

	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	float2 my_position = positions[i];

	float2 a_i = {0.0f, 0.0f};

	// Calculate force from myself to all blocks (Every thread)
	for (int block = 0; block < gridDim.x; block++) {
		int idx = block * blockDim.x + threadIdx.x;
		//Load current batch in
		float2 p = positions[idx];
		sharedposmass[threadIdx.x] = {p.x, p.y, masses[idx], 0.0};

		__syncthreads();

		for (int j = 0; j < blockDim.x; ++j) {
			float4 posmass = sharedposmass[j];
			float2 r_ij = {posmass.x - my_position.x,posmass.y - my_position.y};
			float distSqr = r_ij.x * r_ij.x + r_ij.y * r_ij.y + EPS_2;
			float invDistCube = 1.0f/sqrtf(distSqr * distSqr * distSqr);
			float s = sharedposmass[j].z * invDistCube;
			a_i =  a_i + r_ij * s;
		}
		__syncthreads();
	}

	accelerations[i] = a_i * GRAVITY;
}

__global__ void update(float2 *velocities, float2 *positions, float2 *accelerations, const int n) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	velocities[i] += accelerations[i];
	positions[i] += velocities[i];
}

__global__ void hack(float2 *velocities, float2 *accelerations, float2 *positions, float *masses) {
	positions[0] = make_float2(0, 0);
	masses[0] = 1000.0;
	velocities[0] = make_float2(0, 0);
	accelerations[0] = make_float2(0, 0);
}