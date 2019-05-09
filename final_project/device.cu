#include "device.cuh"


__global__
void partial_derv_gpu(float *dx, float *dy, float *dz, int *pixel, int h, int h2, int width, int height, int depth)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;							// Calc index along X axis
	int idy = threadIdx.y + blockIdx.y * blockDim.y;							// Calc index along Y axis
	int idz = threadIdx.z + blockIdx.z * blockDim.z;							// Calc index along Z axis

	int index = (idz * width * height)+(idy * width)+idx;							// Calculate the overall index as 1D

	if( (idx < width) && (idy < height) && (idz < depth) )							// restricts threads only needed
	{
		if(idx == 0) 			dx[index] = (pixel[index+1] - pixel[index])   / (float)h;	// if first column apply forward diff formula
		else if(idx == width-1) 	dx[index] = (pixel[index]   - pixel[index-1]) / (float)h;	// if last colum apply backward difference formula
		else 				dx[index] = (pixel[index+1] - pixel[index-1]) / (2.0f*h);	// else apply central difference formula

		if(idy == 0) 			dy[index] = (pixel[width+index] - pixel[index])       / (float)h;	// if first column apply forward diff formula
		else if(idy == height-1) 	dy[index] = (pixel[index] 	- pixel[index-width]) / (float)h;	// if first column apply backward diff formula
		else 				dy[index] = (pixel[index+width] - pixel[index-width]) / (2.0f*h);	// if first column apply central diff formula

		if(idz == 0)			dz[index] = (pixel[index+width*height] - pixel[index]) 		    / (float)h2;// if first column apply forward diff formula
		else if(idz == depth - 1)	dz[index] = (pixel[index] 	       - pixel[index-width*height]) / (float)h2;// if first column apply backward diff formula
		else				dz[index] = (pixel[index+width*height] - pixel[index-width*height]) / (2.0f*h2);// if first column apply central diff formula
	}
	else return;
}

__global__
void tensor_gpu(matrix *T, float *dx, float *dy, float *dz, int size)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;							// Calculate the 1D index

	if(idx < size)												// restrict to threads less than size
	{
		float x = dx[idx], y = dy[idx], z = dz[idx];							// copy the data from global to register
		
		T[idx].a1 = x * x;										// Calculate the tensors
		T[idx].a2 = x * y; 
		T[idx].a3 = x * z;
		T[idx].b2 = y * y;
		T[idx].b3 = y * z;
		T[idx].c3 = z * z;
	}
	else return;												// return the unused threads
}

__global__
void padding_gpu(matrix *T, matrix* T_pad, int half, int half_z, int width, int height, int depth)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;							// Calc index along X axis
	int idy = threadIdx.y + blockIdx.y * blockDim.y;							// Calc index along Y axis
	int idz = threadIdx.z + blockIdx.z * blockDim.z;							// Calc index along Z axis

	int index = (idz * width * height)+(idy * width)+idx;							// Calculate overall index as 1D

	if(idx < width && idy < height && idz < depth)								// Check the bounds
	{
		int local_idx, local_idy, local_idz, local_width, local_height, local_index;			// declare local variables for mapping
	        local_idx = idx - half;										// Assign the local variables
                local_idy = idy - half;
                local_idz = idz - half_z;
                local_width = width - 2 * half;
                local_height = height - 2 * half;
                local_index = (local_idz * local_width * local_height) + (local_idy * local_width) + local_idx;	// Calculate the local index
 
                if(idx >= half && idx < (width - half) && idy >= half && idy < (height - half) && idz >= half_z && idz < (depth - half_z))	// Check the bounds apart from halos
                {
                	T_pad[index].a1 = T[local_index].a1;							// Copy the data
                    	T_pad[index].a2 = T[local_index].a2;
                     	T_pad[index].a3 = T[local_index].a3;
                     	T_pad[index].b2 = T[local_index].b2;
                      	T_pad[index].b3 = T[local_index].b3;
                      	T_pad[index].c3 = T[local_index].c3;
           	}
		else												// if halos assign zero's
		{
			T_pad[index].a1 = 0.0f;		
			T_pad[index].a2 = 0.0f;
			T_pad[index].a3 = 0.0f;
			T_pad[index].b2 = 0.0f;
			T_pad[index].b3 = 0.0f;
			T_pad[index].c3 = 0.0f;
		}	
 	}
	else return;												// return unused threads
}

__global__
void filter_x_gpu(matrix *T, matrix *T_x, int filter_size, int filter_size_z, int half, int half_z, int width, int height, int depth)
{
	
	int idx = threadIdx.x + blockIdx.x * blockDim.x;							// Calc index along X axis
	int idy = threadIdx.y + blockIdx.y * blockDim.y;							// Calc index along Y axis
	int idz = threadIdx.z + blockIdx.z * blockDim.z;							// Calc index along Z axis

	extern __shared__ matrix sh_ptr[];									// pointer to shared memory

	if(idx < width && idy < height && idz < depth)								// If threads within bounds copy data to shared memory
	{
		for(int ix = threadIdx.x; ix < (blockDim.x + filter_size - 1); ix += blockDim.x)		// for loop to cover the halo regions along X
		{
			int sh_index = ix + threadIdx.y * (blockDim.x + filter_size - 1) + threadIdx.z * (blockDim.x + filter_size - 1) * blockDim.y;	// compute index of shared memory
			int local_idx = ix + blockIdx.x * blockDim.x;						// correcponding global array index along X
			int local_index = (idz * width * height)+(idy * width)+ local_idx;			// corresponding global array 1D index

			sh_ptr[sh_index] = T[local_index];							// Copy data to shared memory
		}
		__syncthreads();										// synchronize the threads
	}
	if(idx <= (width - filter_size) && idy <= (height - half) && idz <= (depth - half_z))			// make sure the indices stay within needed area
	{
		float temp_a1 = 0.0f, temp_a2 = 0.0f, temp_a3 = 0.0f, temp_b2 = 0.0f, temp_b3 = 0.0f, temp_c3 = 0.0f;	// Initialize temps
		
		int sh_index = threadIdx.x + threadIdx.y * (blockDim.x + filter_size - 1) + threadIdx.z * (blockDim.x + filter_size - 1) * blockDim.y;	// Compute shared index

		for(int ref = 0; ref < filter_size; ref++)							// Iterate throught filter size
		{
			temp_a1 += sh_ptr[sh_index + ref].a1;							// accumulate the sum in temp along x directiom
			temp_a2 += sh_ptr[sh_index + ref].a2;
			temp_a3 += sh_ptr[sh_index + ref].a3;
			temp_b2 += sh_ptr[sh_index + ref].b2;
			temp_b3 += sh_ptr[sh_index + ref].b3;
			temp_c3 += sh_ptr[sh_index + ref].c3;
		}
	       	int local_idx = idx + half;									// Compute the index to which it has to be copied back
            	int local_index = (idz * width * height) + (idy * width) + local_idx;

      		T_x[local_index].a1 = temp_a1 / (float)filter_size;						// normalize the values with filter size
       		T_x[local_index].a2 = temp_a2 / (float)filter_size;
      		T_x[local_index].a3 = temp_a3 / (float)filter_size;
      		T_x[local_index].b2 = temp_b2 / (float)filter_size;
      		T_x[local_index].b3 = temp_b3 / (float)filter_size;
      		T_x[local_index].c3 = temp_c3 / (float)filter_size;
	}
	else return;												// return unused threads
}

/*
__global__
void filter_x_gpu(matrix *T, matrix *T_x, int filter_size, int filter_size_z, int half, int half_z, int width, int height, int depth)
{														// Non shared memory version also works perfectly.
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int idz = threadIdx.z + blockIdx.z * blockDim.z;

	int index = (idz * width * height)+(idy * width)+idx;

	if(idx <= (width - filter_size) && idy <= (height - half) && idz <= (depth - half_z))
	{
		float temp_a1 = 0.0f, temp_a2 = 0.0f, temp_a3 = 0.0f, temp_b2 = 0.0f, temp_b3 = 0.0f, temp_c3 = 0.0f;

              	int local_idx = idx + half;
            	int local_index = (idz * width * height) + (idy * width) + local_idx;

		for(int ref = 0; ref < filter_size; ref++)
              	{
              		temp_a1 += T[index + ref].a1;
                  	temp_a2 += T[index + ref].a2;
                  	temp_a3 += T[index + ref].a3;
                  	temp_b2 += T[index + ref].b2;
               		temp_b3 += T[index + ref].b3;
                	temp_c3 += T[index + ref].c3;
              	}
            	T_x[local_index].a1 = temp_a1 / (float)filter_size;
              	T_x[local_index].a2 = temp_a2 / (float)filter_size;
               	T_x[local_index].a3 = temp_a3 / (float)filter_size;
               	T_x[local_index].b2 = temp_b2 / (float)filter_size;
             	T_x[local_index].b3 = temp_b3 / (float)filter_size;
            	T_x[local_index].c3 = temp_c3 / (float)filter_size;
	}
	else return;
}
*/

__global__
void filter_y_gpu(matrix *T_x, matrix *T_y, int filter_size, int filter_size_z, int half, int half_z, int width, int height, int depth)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;							// Calc index along X axis
	int idy = threadIdx.y + blockIdx.y * blockDim.y;							// Calc index along Y axis
	int idz = threadIdx.z + blockIdx.z * blockDim.z;							// Calc index along Z axis

	extern __shared__ matrix sh_ptr[];									// pointer to shared memory

	if(idx < width && idy < height && idz < depth)								// If threads within bounds copy data to shared memory
	{
		for(int iy = threadIdx.y; iy < (blockDim.y + filter_size - 1); iy += blockDim.y)		// for loop to cover the halo regions along Y
		{
			int sh_index = threadIdx.x + iy * blockDim.x + threadIdx.z * blockDim.x * (blockDim.y + filter_size - 1);	// compute index of shared memory
			int local_idy = iy + blockIdx.y * blockDim.y;						// correcponding global array index along Y
			int local_index = (idz * width * height)+(local_idy * width)+ idx;			// corresponding global array 1D index

			sh_ptr[sh_index] = T_x[local_index];							// Copy data to shared memory
		}
		__syncthreads();										// synchronize the threads
	}

	if(idx <= (width - half) && idy <= (height - filter_size) && idz <= (depth - half_z))			// make sure the indices stay within needed area
	{

		float temp_a1 = 0.0f, temp_a2 = 0.0f, temp_a3 = 0.0f, temp_b2 = 0.0f, temp_b3 = 0.0f, temp_c3 = 0.0f;// Initialize temps

	 	int sh_index = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * (blockDim.y + filter_size - 1);// Compute shared index


		int local_width = (blockDim.x);
		for(int ref = 0; ref < filter_size; ref++)							// Iterate throught filter size
		{
			temp_a1 += sh_ptr[sh_index + (ref * local_width)].a1;					// accumulate the sum in temp along Y directiom
			temp_a2 += sh_ptr[sh_index + (ref * local_width)].a2;
			temp_a3 += sh_ptr[sh_index + (ref * local_width)].a3;
			temp_b2 += sh_ptr[sh_index + (ref * local_width)].b2;
			temp_b3 += sh_ptr[sh_index + (ref * local_width)].b3;
			temp_c3 += sh_ptr[sh_index + (ref * local_width)].c3;
		}

              	int local_idy = idy + half;									// Compute the index to which it has to be copied back
            	int local_index = (idz * width * height) + (local_idy * width) + idx;

          	T_y[local_index].a1 = temp_a1 / (float)filter_size;						// normalize the values with filter size
          	T_y[local_index].a2 = temp_a2 / (float)filter_size;
           	T_y[local_index].a3 = temp_a3 / (float)filter_size;
           	T_y[local_index].b2 = temp_b2 / (float)filter_size;
             	T_y[local_index].b3 = temp_b3 / (float)filter_size;
          	T_y[local_index].c3 = temp_c3 / (float)filter_size;
	}
	else return;												// return threads if unused
}

/*
__global__
void filter_y_gpu(matrix *T_x, matrix *T_y, int filter_size, int filter_size_z, int half, int half_z, int width, int height, int depth)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;							// NON shared memory version works good
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int idz = threadIdx.z + blockIdx.z * blockDim.z;

	int index = (idz * width * height)+(idy * width)+idx;


	if(idx <= (width - half) && idy <= (height - filter_size) && idz <= (depth - half_z))
	{
		float temp_a1 = 0.0f, temp_a2 = 0.0f, temp_a3 = 0.0f, temp_b2 = 0.0f, temp_b3 = 0.0f, temp_c3 = 0.0f;

              	int local_idy = idy + half;
            	int local_index = (idz * width * height) + (local_idy * width) + idx;

		for(int ref = 0; ref < filter_size; ref++)
              	{
            		temp_a1 += T_x[index + (ref * width)].a1;
                 	temp_a2 += T_x[index + (ref * width)].a2;
                  	temp_a3 += T_x[index + (ref * width)].a3;
                      	temp_b2 += T_x[index + (ref * width)].b2;
                    	temp_b3 += T_x[index + (ref * width)].b3;
                  	temp_c3 += T_x[index + (ref * width)].c3;
           	}
           	T_y[local_index].a1 = temp_a1 / filter_size;
          	T_y[local_index].a2 = temp_a2 / filter_size;
           	T_y[local_index].a3 = temp_a3 / filter_size;
           	T_y[local_index].b2 = temp_b2 / filter_size;
             	T_y[local_index].b3 = temp_b3 / filter_size;
          	T_y[local_index].c3 = temp_c3 / filter_size;
	}
	else return;
}
*/

__global__
void filter_z_gpu(matrix *T_y, matrix *T_z, int filter_size, int filter_size_z, int half, int half_z, int width, int height, int depth)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;							// Calc X index
	int idy = threadIdx.y + blockIdx.y * blockDim.y;							// Calc Y index
	int idz = threadIdx.z + blockIdx.z * blockDim.z;							// Calc Z index

	extern __shared__ matrix sh_ptr[];									// pointer to shared memory

	if(idx < width && idy < height && idz < depth)								// check if threads within bounds
	{
		for(int iz = threadIdx.z; iz < (blockDim.z + filter_size_z - 1); iz += blockDim.z)		// iterate through threads along z
		{
			
			int sh_index = threadIdx.x + (threadIdx.y * blockDim.x) + (iz * blockDim.x * blockDim.y);	// compute shared mem index
			int local_idz = iz + blockIdx.z * blockDim.z;						// compute the index from which the data has to be fetched
			int local_index = (local_idz * width * height)+(idy * width)+ idx;
			if(local_idz < depth)									// causes illegal memory acces without this
			{
				sh_ptr[sh_index] = T_y[local_index];						// copy data to shared memory
			}
		}
		__syncthreads();										// Sync the threads within blocks
	}

	if(idx <= (width - half) && idy <= (height - half) && idz <= (depth - filter_size_z))			// make sure the indices stay within needed area
	{
		float temp_a1 = 0.0f, temp_a2 = 0.0f, temp_a3 = 0.0f, temp_b2 = 0.0f, temp_b3 = 0.0f, temp_c3 = 0.0f;// Initialize temps

	 	int sh_index = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;	// Compute shared index

		for(int ref = 0; ref < filter_size_z; ref++)							// Iterate throught filter size
              	{
            		temp_a1 += sh_ptr[sh_index + (ref * blockDim.x * blockDim.y)].a1;			// accumulate the sum in temp along Z directiom
                 	temp_a2 += sh_ptr[sh_index + (ref * blockDim.x * blockDim.y)].a2;
                  	temp_a3 += sh_ptr[sh_index + (ref * blockDim.x * blockDim.y)].a3;
                      	temp_b2 += sh_ptr[sh_index + (ref * blockDim.x * blockDim.y)].b2;
                    	temp_b3 += sh_ptr[sh_index + (ref * blockDim.x * blockDim.y)].b3;
                  	temp_c3 += sh_ptr[sh_index + (ref * blockDim.x * blockDim.y)].c3;
           	}

               	int local_idz = idz + half_z;									// Compute the index to which it has to be copied back
            	int local_index = (local_idz * width * height) + (idy * width) + idx;

          	T_z[local_index].a1 = temp_a1 / filter_size_z;							// normalize the values
          	T_z[local_index].a2 = temp_a2 / filter_size_z;
           	T_z[local_index].a3 = temp_a3 / filter_size_z;
           	T_z[local_index].b2 = temp_b2 / filter_size_z;
             	T_z[local_index].b3 = temp_b3 / filter_size_z;
          	T_z[local_index].c3 = temp_c3 / filter_size_z;
	}
	else return;												// return unused threads
}

/*
__global__
void filter_z_gpu(matrix *T_y, matrix *T_z, int filter_size, int filter_size_z, int half, int half_z, int width, int height, int depth)
{														// NON shared memory version works well
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int idz = threadIdx.z + blockIdx.z * blockDim.z;

	int index = (idz * width * height)+(idy * width)+idx;


	if(idx <= (width - half) && idy <= (height - half) && idz <= (depth - filter_size_z))
	{
		float temp_a1 = 0, temp_a2 = 0, temp_a3 = 0, temp_b2 = 0, temp_b3 = 0, temp_c3 = 0;

              	int local_idz = idz + half_z;
            	int local_index = (local_idz * width * height) + (idy * width) + idx;

		for(int ref = 0; ref < filter_size_z; ref++)
              	{
            		temp_a1 += T_y[index + (ref * width * height)].a1;
                 	temp_a2 += T_y[index + (ref * width * height)].a2;
                  	temp_a3 += T_y[index + (ref * width * height)].a3;
                      	temp_b2 += T_y[index + (ref * width * height)].b2;
                    	temp_b3 += T_y[index + (ref * width * height)].b3;
                  	temp_c3 += T_y[index + (ref * width * height)].c3;
           	}
           	T_z[local_index].a1 = temp_a1 / filter_size_z;
          	T_z[local_index].a2 = temp_a2 / filter_size_z;
           	T_z[local_index].a3 = temp_a3 / filter_size_z;
           	T_z[local_index].b2 = temp_b2 / filter_size_z;
             	T_z[local_index].b3 = temp_b3 / filter_size_z;
          	T_z[local_index].c3 = temp_c3 / filter_size_z;
	}
	else return;
}
*/

__global__
void remove_padding_gpu(matrix *T_z, matrix *T_m, int half, int half_z, int width, int height, int depth)
{

	int idx = threadIdx.x + blockIdx.x * blockDim.x;							// Calc X index
	int idy = threadIdx.y + blockIdx.y * blockDim.y;							// Calc Y index
	int idz = threadIdx.z + blockIdx.z * blockDim.z;							// Calc Z index

	int index = (idz * width * height)+(idy * width)+idx;							// Calc Overall index

	if(idx >= half && idx < (width - half) && idy >= half && idy < (height - half) && idz >= half_z && idz < (depth - half_z))	// use only necessary threads
	{
		int local_idx = idx - half;									// calculate the indices to be copied to
		int local_idy = idy - half;
		int local_idz = idz - half_z;
		int local_width = width - 2 * half;
		int local_height = height - 2 * half;
		int local_index = (local_idz * local_width * local_height) + (local_idy * local_width) + local_idx;	

		T_m[local_index].a1 = T_z[index].a1;								// copy the data
		T_m[local_index].a2 = T_z[index].a2;
		T_m[local_index].a3 = T_z[index].a3;
		T_m[local_index].b2 = T_z[index].b2;
		T_m[local_index].b3 = T_z[index].b3;
		T_m[local_index].c3 = T_z[index].c3;
	}
	else return;												// return unused threads
}

__device__
void eulers_method(position *trace, eigen_vectors *Evec, int idx, int idy, int idz, int width, int height, int depth, float del_t, int n)
{
	position initial = {(float) idx, (float) idy, (float) idz};						// Assign the initial position
	position new_pos = {0.0f, 0.0f, 0.0f};									// declare new position

	int init;												// declare index for initial

	trace[0].x = initial.x;	trace[0].y = initial.y;	trace[0].z = initial.z;					// copy initial to trace

	int step = 0;												// Initialize the step
	while(step < n)												// iterate n times
	{	
		init   = ((int)initial.z * width * height) + ((int)initial.y * width) + ((int)initial.x);	// calculate initial index

		new_pos.x = initial.x + del_t * (Evec[init].x);							// Calculate new pos using eulers method	
		new_pos.y = initial.y + del_t * (Evec[init].y);
		new_pos.z = initial.z + del_t * (Evec[init].z);

		if( (new_pos.x < width) && (new_pos.y < height) && (new_pos.z < depth) && (new_pos.x >= 0) && (new_pos.y >= 0) && (new_pos.z >= 0) && \
		    (initial.x < width) && (initial.y < height) && (initial.z < depth) && (initial.x >= 0) && (initial.y >= 0) && (initial.z >= 0) 	)
														// only if indices stay within bounds
		{


			int new_p  = ((int)new_pos.z * width * height) + ((int)new_pos.y * width) + ((int)new_pos.x);	// calc index of new position

			float dot_p = (Evec[new_p].x * Evec[init].x) + (Evec[new_p].y * Evec[init].y) + (Evec[new_p].z * Evec[init].z);	// calculate the dot product of 2 vectors
			if(dot_p < 0)										// if negative
			{
				new_pos.x = initial.x + del_t * (-Evec[init].x);				// change direction of the vector and calculate new position
				new_pos.y = initial.y + del_t * (-Evec[init].y);
				new_pos.z = initial.z + del_t * (-Evec[init].z);
			}
			
			step++;											// increment step
			
			trace[step].x = new_pos.x;								// copy new position to trace
			trace[step].y = new_pos.y;
			trace[step].z = new_pos.z;

			initial.x = new_pos.x;									// copy new position to initial
			initial.y = new_pos.y;
			initial.z = new_pos.z;
		}
		else												// if not within bounds
		{
			step++;											// increment step
			trace[step].x = -1;									// assign data not within bound
			trace[step].y = -1;
			trace[step].z = -1;
		}
	}
	return;												
}

__global__
void tractography_gpu(position *trace, eigen_vectors *Evec, float del_t, int n_steps, int width, int height, int depth)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;							// Calc X indices
	int idy = threadIdx.y + blockIdx.y * blockDim.y;							// Calc Y indices
	int idz = threadIdx.z + blockIdx.z * blockDim.z;							// Calc Z indices

	int index = (idz * width * height)+(idy * width)+idx;							// Calculate overall index as 1D
	
	if(idx < width && idy < height && idz < depth)								// if within bounds
	{
		eulers_method(&trace[index*(n_steps+1)], Evec, idx, idy, idz, width, height, depth, del_t, n_steps);	// call the device function 
	}
	else return;												// return unused threads
}

void partial_derv(float *dx, float *dy, float *dz, int *pixel, int h, int h2, int width, int height, int depth)
{
	dim3 threads(8,8,8);
	dim3 blocks(ceil(width/8.0f), ceil(height/8.0f), ceil(depth/8.0f));
	partial_derv_gpu<<<blocks, threads>>>(dx, dy, dz, pixel, h, h2, width, height, depth);
}


void tensor(matrix *T, float *dx, float *dy, float *dz, int size)
{
	dim3 threads(1024);
	dim3 blocks(ceil(size/1024.0));
	tensor_gpu<<<blocks, threads>>> (T, dx, dy, dz, size);
}

void padding(matrix *T,matrix* T_pad, int half, int half_z, int width, int height, int depth)
{
	dim3 threads(8,8,8);
	dim3 blocks(ceil(width/8.0f), ceil(height/8.0f), ceil(depth/8.0f));
	padding_gpu<<<blocks, threads>>>(T, T_pad, half, half_z, width, height, depth);	
}

void filter_x(matrix *T, matrix *T_x, int filter_size, int filter_size_z, int half, int half_z, int width, int height, int depth)
{
	dim3 threads(8,8,8);
	dim3 blocks(ceil(width/8.0f), ceil(height/8.0f), ceil(depth/8.0f));
	size_t sh_mem_size = (8 + filter_size - 1) * 8 * 8 * sizeof(matrix); 
	filter_x_gpu<<<blocks, threads, sh_mem_size>>>(T, T_x, filter_size, filter_size_z, half, half_z, width, height, depth);	
}

void filter_y(matrix *T_x, matrix *T_y, int filter_size, int filter_size_z, int half, int half_z, int width, int height, int depth)
{
	dim3 threads(8,8,8);
	dim3 blocks(ceil(width/8.0f), ceil(height/8.0f), ceil(depth/8.0f));
	size_t sh_mem_size = 8 * (8 + filter_size) * 8 * sizeof(matrix); 
	filter_y_gpu<<<blocks, threads, sh_mem_size>>>(T_x, T_y, filter_size, filter_size_z, half, half_z, width, height, depth);	
}

void filter_z(matrix *T_y, matrix *T_z, int filter_size, int filter_size_z, int half, int half_z, int width, int height, int depth)
{
	dim3 threads(8,8,8);
	dim3 blocks(ceil(width/8.0f), ceil(height/8.0f), ceil(depth/8.0f));
	size_t sh_mem_size = 8 * 8 * (8 + filter_size_z - 1) * sizeof(matrix); 
	filter_z_gpu<<<blocks, threads, sh_mem_size>>>(T_y, T_z, filter_size, filter_size_z, half, half_z, width, height, depth);	
}

void remove_padding(matrix *T_z, matrix *T_m, int half, int half_z, int width, int height, int depth)
{
	dim3 threads(8,8,8);
	dim3 blocks(ceil(width/8.0f), ceil(height/8.0f), ceil(depth/8.0f));
	remove_padding_gpu<<<blocks, threads>>>(T_z, T_m, half, half_z, width, height, depth);	
}

void tractography(position *trace, eigen_vectors *Evec, float del_t, int n_steps, int width, int height, int depth)
{
	dim3 threads(8,8,8);
	dim3 blocks(ceil(width/8.0f), ceil(height/8.0f), ceil(depth/8.0f));
	tractography_gpu<<<blocks, threads>>>(trace, Evec, del_t, n_steps, width, height, depth);	
}
