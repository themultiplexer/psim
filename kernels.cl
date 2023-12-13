#define EPS_2 0.00001f
#define GRAVITY 0.00000001f

__kernel void applyForces(__global float2 *accelerations, __global float2 *positions, __local float4 sharedposmass[], __global float *masses) {
	unsigned int i = get_local_size(0) * get_group_id(0) + get_local_id(0);
	float2 my_position = positions[i];

	float2 a_i = (float2)(0.0f, 0.0f);

	// Calculate force from myself to all blocks (Every thread)
	for (int block = 0; block < get_num_groups(0); block++) {
		int idx = block * get_local_size(0) + get_local_id(0);
		//Load current batch in
		float2 p = positions[idx];
		sharedposmass[get_local_id(0)] = (float4)(p.x, p.y, masses[idx], 0.0);

		barrier(CLK_LOCAL_MEM_FENCE);

		for (int j = 0; j < get_local_size(0); ++j) {
			float4 posmass = sharedposmass[j];
			float2 r_ij = {posmass.x - my_position.x,posmass.y - my_position.y};
			float distSqr = r_ij.x * r_ij.x + r_ij.y * r_ij.y + EPS_2;
			float invDistCube = 1.0f/sqrt(distSqr * distSqr * distSqr);
			float s = sharedposmass[j].z * invDistCube;
			a_i =  a_i + r_ij * s;
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	accelerations[i] = a_i * GRAVITY;
}

__kernel void update(__global float2 *velocities, __global float2 *positions, __global float2 *accelerations) {
	unsigned int i = get_local_size(0) * get_group_id(0) + get_local_id(0);
	velocities[i] += accelerations[i];
	positions[i] += velocities[i];
}

__kernel void hack(__global float2 *velocities, __global float2 *accelerations, __global float2 *positions, __global float *masses) {
	positions[0] = (float2)(0, 0);
	masses[0] = 1000.0;
	velocities[0] = (float2)(0, 0);
	accelerations[0] = (float2)(0, 0);
}