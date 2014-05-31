#include <cstdio>
#include <cstdlib>
#include <ctime>

#define DO_STATS
#define M_SIZE		128
typedef struct {
	int births;
	int deaths;
	int alive;
	int dead;
} MapStats;

template <size_t xSize, size_t ySize, size_t zSize>
void initMap(unsigned char *oldMap, int mapSeed);
template <size_t xSize, size_t ySize, size_t zSize>
void printJSON(unsigned char *map, int iter);
template <size_t xSize, size_t ySize, size_t zSize>
void countStats(unsigned char *oldMap, unsigned char *newMap, MapStats &stats);

//On Device function
template <size_t xSize, size_t ySize, size_t zSize>
__device__ int countNeighbours(unsigned char *map, int x, int y, int z);
//Kernels
template<size_t xSize, size_t ySize, size_t zSize>
__global__ void unopIterate(unsigned char *d_oldMap, unsigned char *d_newMap, int iters, int bLim, int dLim);
template<size_t xSize, size_t ySize, size_t zSize>
__global__ void opIterate(unsigned char *d_oldMap, unsigned char *d_newMap, int iters);


//Globals -- bad code
const int deathLimit = 15;
const int birthLimit = 17;

int main()
{
	const int mapSeed = 45000;
	const int xSize = M_SIZE;
	const int ySize = M_SIZE;
	const int zSize = M_SIZE;
	const int maxIters = 30;
	const bool PRINT = false;
	const bool TIME = true;
	
	const int mapSize = xSize*ySize*zSize;
	
	const int blockSize = 8;
	dim3 blockDim(blockSize, blockSize, blockSize);
	dim3 gridDim(xSize/blockSize, ySize/blockSize, zSize/blockSize);
	
	clock_t start, total = 0;
	unsigned char *oldMap = new unsigned char[xSize*ySize*zSize];
	unsigned char *newMap = new unsigned char[xSize*ySize*zSize];
	unsigned char *temp;
	MapStats stats;
	
	unsigned char *d_oldMap;
	unsigned char *d_newMap;
	
	
	cudaMalloc((void **) &d_oldMap, mapSize);
	cudaMalloc((void **) &d_newMap, mapSize);
	
	initMap<xSize,ySize,zSize>(oldMap, mapSeed);
	
	if(PRINT)
	{
		printf("{\n");
		printf("\"mapSeed\":%d,\n", mapSeed);
		printf("\"deathLimit\":%d,\n", deathLimit);
		printf("\"birthLimit\":%d,\n", birthLimit);
		printf("\"xSize\":%d,\n", xSize);
		printf("\"ySize\":%d,\n", ySize);
		printf("\"zSize\":%d,\n", zSize);
		printf("\"maxIters\":%d,\n", maxIters);
		printf("\"mapData\" : [\n");
		printJSON<xSize,ySize,zSize>(oldMap, 0);	//Iteration 0 is starting iteration
	}
	
	for(int iter=0; iter<maxIters; ++iter)
	{
		if(PRINT) printf(",\n");
		
		
		//Main iteration section
		if(TIME) start = clock();
		cudaMemcpy(d_oldMap, oldMap, mapSize, cudaMemcpyHostToDevice);
		unopIterate<xSize,ySize,zSize><<<gridDim,blockDim>>>(d_oldMap, d_newMap, 1, birthLimit, deathLimit);
		cudaMemcpy(newMap, d_newMap, mapSize, cudaMemcpyDeviceToHost);
		if(TIME) total += clock() - start;
		
		#ifdef DO_STATS
		if(iter == maxIters-1)
		{
			countStats<xSize,ySize,zSize>(oldMap, newMap, stats);
			printf("[%d] ",iter+1);
			printf("births: %d   \tdeaths: %d   \talive: %d   \tdead: %d   \ttotal: %d\n", stats.births, stats.deaths, stats.alive, stats.dead, stats.alive+stats.dead);
		}
		#endif
		if(PRINT) printJSON<xSize,ySize,zSize>(newMap, iter+1);
		temp = oldMap;
		oldMap = newMap;
		newMap = temp;
	}
	if(PRINT) printf("\n]}\n");
	if(TIME)
	{
		double diff = (double(total))/CLOCKS_PER_SEC;
		printf("time: took %f seconds for %dx%dx%d matrix\n", diff, xSize, ySize, ySize);
	}
	delete[] oldMap;
	delete[] newMap;
	
	return 0;
}

template <size_t xSize, size_t ySize, size_t zSize>
void initMap(unsigned char *oldMap, int mapSeed)
{
	srand(mapSeed);
	for(int k=0; k<zSize; ++k)
	{
		for(int j=0; j<ySize; ++j)
		{
			for(int i=0; i<xSize; ++i)
			{
				oldMap[k*(xSize*ySize)+j*xSize+i] = rand() % 2;
			}
		}
	}
}

//Count neighbours that are alive
template <size_t xSize, size_t ySize, size_t zSize>
__device__ int countNeighbours(unsigned char *map, int x, int y, int z)
{
	const bool countBounds = true;
    int count = 0;
    for(int k=-1; k<2; ++k)
	{
        for(int j=-1; j<2; ++j)
		{
			for(int i=-1; i<2; ++i)
			{
				//Count all except middle point
				if( i != 0 || j != 0 || k != 0)
				{
					int xPos = x + i;
					int yPos = y + j;
					int zPos = z + k;
					
					//Check boundaries
					if(xPos < 0 || yPos < 0 || zPos < 0 || xPos >= xSize || yPos >= ySize || zPos >= zSize)
					{
						//if(x==0 && y==0 && z==0) printf("(%d,%d,%d):bounds\n",xPos,yPos,zPos);
						if(countBounds) count++;
					}
					else 
					{
						//if(x==0 && y==0 && z==0) printf("(%d,%d,%d):not bounds\n",xPos,yPos,zPos);
						count += map[zPos*(xSize*ySize)+yPos*xSize+xPos];
					}
				}
			}
        }
    }
	return count;
}

template<size_t xSize, size_t ySize, size_t zSize>
__global__ void unopIterate(unsigned char *d_oldMap, unsigned char *d_newMap, int iters, int bLim, int dLim)
{
	const int globalx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int globaly = (blockIdx.y * blockDim.y) + threadIdx.y;
	const int globalz = (blockIdx.z * blockDim.z) + threadIdx.z;
	
	//Only perform action if thread is inside the bounds of the grid
	if( !(globalx >= xSize || globaly >= ySize || globalz >= zSize) )
	{
		int globalIndex = globalz*(xSize * ySize) + globaly*(xSize)+globalx;
		int aliveCnt = countNeighbours<xSize,ySize,zSize>(d_oldMap, globalx, globaly, globalz);
		if(d_oldMap[globalIndex] == 1) 
		{
			d_newMap[globalIndex] = (aliveCnt < dLim) ? 0 : 1;
		}
		else
		{
			d_newMap[globalIndex] = (aliveCnt > bLim) ? 1 : 0;
		}
	}
}

template<size_t xSize, size_t ySize, size_t zSize>
__global__ void opIterate(unsigned char *d_oldMap, unsigned char *d_newMap, int iters)
{
	const bool countBounds = true;

	//Calculate size of memory block and pad each dimension by + 2
	const int s_xSize = blockDim.x + 2;
	const int s_ySize = blockDim.y + 2;
	const int s_zSize = blockDim.z + 2;
	const int arrSize = s_xSize * s_ySize * s_zSize;

	const int localx = threadIdx.x + 1;
	const int localy = threadIdx.y + 1;
	const int localz = threadIdx.z + 1;

	const int globalx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int globaly = (blockIdx.y * blockDim.y) + threadIdx.y;
	const int globalz = (blockIdx.z * blockDim.z) + threadIdx.z;

	//Create shared memory block, pass in allocation for arrSize*2
	extern __shared__ unsigned char tmp[];

	//Split memory block into real units
	unsigned char *s_oldMap = tmp;
	unsigned char *s_newMap = &tmp[arrSize];

	//Only perform action if thread is inside the bounds of the grid
	if( !(globalx >= xSize || globaly >= ySize || globalz >= zSize) )
	{
		//Populate shared old map data with global oldmap data
		//Handle edge cases
		if(localx == 1)
		{
			const int localIndex = localz*(s_xSize*s_ySize)+localy*(s_xSize);
			const int globalIndex = globalz*(xSize*ySize)+globaly*(xSize)+(globalx-1);
			if(0 <= globalIndex)
			{
				s_oldMap[localIndex]  = d_oldMap[globalIndex];
			}
			else
			{
				s_oldMap[localIndex] = (countBounds) ? 1 : 0;
			}
		}

		if(localy == 1)
		{
			const int localIndex = localz*(s_xSize*s_ySize)+localx;
			const int globalIndex  = globalz*(xSize*ySize)+(globaly-1)*(xSize)+globalx;
			if(0 <= globalIndex)
			{
				s_oldMap[localIndex]  = d_oldMap[globalIndex];
			}
			else
			{
				s_oldMap[localIndex] = (countBounds) ? 1 : 0;
			}
		}

		if(localz == 1)
		{
			const int localIndex = localy*(s_xSize)+localx;
			const int globalIndex = (globalz-1)*(xSize*ySize)+globaly*(xSize)+globalx;
			if(0 <= globalIndex)
			{
				s_oldMap[localIndex]  = d_oldMap[globalIndex];
			}
			else
			{
				s_oldMap[localIndex] = (countBounds) ? 1 : 0;
			}
		}

		if(localx == 1 && localy == 1)
		{
			const int localIndex = localz*(s_xSize*s_ySize);
			const int globalIndex = globalz*(xSize*ySize)+(globaly-1)*(xSize)+(globalx-1);
			if(0 <= globalIndex)
			{
				s_oldMap[localIndex]  = d_oldMap[globalIndex];
			}
			else
			{
				s_oldMap[localIndex] = (countBounds) ? 1 : 0;
			}
		}

		if(localx == 1 && localz == 1)
		{
			const int localIndex = localy*(s_xSize);
			const int globalIndex = (globalz-1)*(xSize*ySize)+globaly*(xSize)+(globalx-1);
			if(0 <= globalIndex)
			{
				s_oldMap[localIndex]  = d_oldMap[globalIndex];
			}
			else
			{
				s_oldMap[localIndex] = (countBounds) ? 1 : 0;
			}
		}

		if(localy == 1 && localz == 1)
		{
			const int localIndex = localx;
			const int globalIndex = (globalz-1)*(xSize*ySize)+(globaly-1)*(xSize)+globalx;
			if(0 <= globalIndex)
			{
				s_oldMap[localIndex]  = d_oldMap[globalIndex];
			}
			else
			{
				s_oldMap[localIndex] = (countBounds) ? 1 : 0;
			}
		}

		if(localx == 1 && localy == 1 && localz == 1)
		{
			const int localIndex = 0;
			const int globalIndex = (globalz-1)*(xSize*ySize)+(globaly-1)*(xSize)+(globalx-1);
			if(0 <= globalIndex)
			{
				s_oldMap[localIndex]  = d_oldMap[globalIndex];
			}
			else
			{
				s_oldMap[localIndex] = (countBounds) ? 1 : 0;
			}
		}

		if(localx == blockDim.x)
		{
		}

		if(localy == blockDim.y)
		{
		}

		if(localz == blockDim.z)
		{
		}
		//Get normal case
		s_oldMap[localz*(s_xSize*s_ySize)+localy*(s_xSize)+localx] = d_oldMap[globalz*(xSize*ySize)+globaly*(xSize)+globalx];

		__syncthreads();

		//Perform iteration, putting result in s_newMap
		/*
		for(int k=0; k<zSize; ++k)
		{
			for(int j=0; j<ySize; ++j)
			{
				for(int i=0; i<xSize; ++i)
				{
					int aliveCnt = countNeighbours<xSize,ySize,zSize>(oldMap, i, j, k);
					//if(aliveCnt > highcount) highcount = aliveCnt;
					//If cell is alive, check if it dies
					if(oldMap[k*(xSize*ySize)+j*xSize+i] == 1)
					{
						newMap[k*(xSize*ySize)+j*xSize+i] = (aliveCnt < deathLimit) ? 0 : 1;
						#ifdef DO_STATS
						if(aliveCnt < deathLimit)
						{
							++deaths;
							++dead;
						}
						else ++alive;	//still living, yolo
						#endif
					}
					else
					{
						newMap[k*(xSize*ySize)+j*xSize+i] = (aliveCnt > birthLimit) ? 1 : 0;
						#ifdef DO_STATS
						if(aliveCnt > birthLimit)
						{
							++births;
							++alive;
						}
						else ++dead;	//still dead
						#endif
					}
				}
			}
		}
		*/
		__syncthreads();
	//Push new edges into matrix
	}
	
}

template <size_t xSize, size_t ySize, size_t zSize>
void printJSON(unsigned char *map, int iter)
{
	printf("\t{\n");
	printf("\t\"iteration\" : %d,\n", iter);
	printf("\t\"map\" : [\n");
	for(int i=0; i<xSize; ++i)
	{
		for(int j=0; j<ySize; ++j)
		{
			for(int k=0; k<zSize; ++k)
			{
				if( !(i == 0 && j == 0 && k == 0) ) printf(",\n");
				char *val = ( map[k*(xSize*ySize)+j*xSize+i] ) ? "true" : "false";
				printf("\t\t{\"x\":%d, \"y\":%d, \"z\":%d, \"value\":%s}", i, j, k, val);
			}
		}
	}
	printf("\n\t\t]\n\t}");
}

template <size_t xSize, size_t ySize, size_t zSize>
void countStats(unsigned char *oldMap, unsigned char *newMap, MapStats &stats)
{
	int oldAlive = 0, oldDead = 0;
	int newAlive = 0, newDead = 0;
	
	for(int k=0; k<zSize; ++k)
	{
		for(int j=0; j<ySize; ++j)
		{
			for(int i=0; i<xSize; ++i)
			{
				oldAlive += oldMap[k*(xSize*ySize)+j*xSize+i];
				newAlive += newMap[k*(xSize*ySize)+j*xSize+i];
			}
		}
	}
	stats.alive = newAlive;
	newDead= (xSize*ySize*zSize) - newAlive;
	oldDead = (xSize*ySize*zSize) - oldAlive;
	stats.dead = newDead;
	stats.births = (newAlive > oldAlive) ? newAlive - oldAlive : 0;
	stats.deaths = (newDead > oldDead) ? newDead - oldDead : 0;
}


