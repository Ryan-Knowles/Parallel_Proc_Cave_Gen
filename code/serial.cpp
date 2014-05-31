#include <cstdio>
#include <cstdlib>
#include <ctime>

#define DO_STATS
#define M_SIZE		768

typedef struct {
	int births;
	int deaths;
	int alive;
	int dead;
} MapStats;

template <size_t xSize, size_t ySize, size_t zSize>
void initMap(unsigned char *oldMap, int mapSeed);
template <size_t xSize, size_t ySize, size_t zSize>
void iterateCA(unsigned char *oldMap, unsigned char *newMap);
template <size_t xSize, size_t ySize, size_t zSize>
int countNeighbours(unsigned char *map, int x, int y, int z);
template <size_t xSize, size_t ySize, size_t zSize>
void printJSON(unsigned char *map, int iter);
template <size_t xSize, size_t ySize, size_t zSize>
void countStats(unsigned char *oldMap, unsigned char *newMap, MapStats &stats);

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
	
	clock_t start, total = 0;
	unsigned char *oldMap = new unsigned char[xSize*ySize*zSize];
	unsigned char *newMap = new unsigned char[xSize*ySize*zSize];
	initMap<xSize,ySize,zSize>(oldMap, mapSeed);
	MapStats stats;
	
	unsigned char *temp;
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
		
		
		//Start iteration section
		if(TIME) start = clock();
		iterateCA<xSize,ySize,zSize>(oldMap, newMap);
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

template <size_t xSize, size_t ySize, size_t zSize>
void iterateCA(unsigned char *oldMap, unsigned char *newMap)
{
	for(int k=0; k<zSize; ++k)
	{
		for(int j=0; j<ySize; ++j)
		{
			for(int i=0; i<xSize; ++i)
			{
				int aliveCnt = countNeighbours<xSize,ySize,zSize>(oldMap, i, j, k);
				//If cell is alive, check if it dies
				if(oldMap[k*(xSize*ySize)+j*xSize+i] == 1)
				{
					newMap[k*(xSize*ySize)+j*xSize+i] = (aliveCnt < deathLimit) ? 0 : 1;
				}
				else
				{
					newMap[k*(xSize*ySize)+j*xSize+i] = (aliveCnt > birthLimit) ? 1 : 0;
				}
				//printf("(%d,%d,%d): %d->%d [%d]\n",i,j,k,oldMap[i+j*xSize+k*(xSize*ySize)],newMap[i+j*xSize+k*(xSize*ySize)],aliveCnt);
			}
		}
	}
	
}

//Count neighbours that are alive
template <size_t xSize, size_t ySize, size_t zSize>
int countNeighbours(unsigned char *map, int x, int y, int z)
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
				char *val = ( map[i+j*xSize+k*(xSize*ySize)] ) ? "true" : "false";
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
