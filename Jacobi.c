/*
	IP credit ~ Argonne National Laboratory

	Isaac Lehman ~ COMP322 ~ Laplace Heat Distribution

		- openMP + MPIversion

	Dr. Valentines project description:
	Laplace’s Equation can be used to determine the final steady state 
	temperature of a 2D plate by solving (with the appropriate boundary conditions)

	A numerical method derived to solve this problem iteratively is known as Jacobi Iteration. 
	We will start with the demo code from Argonne National Lab.  It works as follows:
		1.	Divide the 2D plate into an appropriate number of “points” (ie cells in a 2D array)

		2.	Initialize all the points to appropriate initial temperatures. In our solution, 
		the four edges will be set to and constantly held at user-selected boundary values; 
		the interior points will be set to the average of the 4 boundary values.  For example, 
		a system with North & South set to 100, and East and West set to 20, would have 
		all the interior points initialized to (100+100+20+20)/4, or 60.

		3. 	Using a second array (plate), generate the “next” temperature of each internal cell as:
				newAry[r][c] = (ary[r-1][c] + ary[r+1][c] + ary[r][c+1] + ary[r][c-1]) / 4.0
				
				That is the next temperature in this cell will be the average of the 4 current 
				neighboring cells (North, South, East, West).  
				
				NB: the perimeter cells of the array are taken to be constant, 
				unchanging values (boundary values).  Only the interior points change value.

		4.	Add the square of all the differences between oldCells & newCells (diffNorm)
							diffNorm=SUM{oldCell-newCell}^2

		5.	Swap the two plates so all your newly calculated values become the current values. 
			The former ary is now the destination for the next cycle of calculations.

		6.	Continue this array upgrade process until the square root of diffNorm is smaller 
			than some EPSILON value OR the total number of iterations exceeds some MAX value.
	
	At the end of each cycle, swap the two arrays

	-----------------------------------------------------------------------------------------


	We chose to use floats rather than doubles since we don't need the
	extra precision and they take half the storage


	This program uses hybrid parallelism with distributed and shared memory.  

	REMEMBER: match tags with MPI send and recieves
*/


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <string.h>
#include <mpi.h>

// debug mode has all the extra print statements...
// run mode just prints out the final statistics for easy piping to a file
#define DEBUG 1 // 1 for debug mode, 0 for run mode

// constant temperatures
#define NORTH 100 
#define SOUTH 100
#define EAST 0
#define WEST 0

#define MAX_TEMP 100 // the max of N, S, E, W
#define MIN_TEMP 0 // the min of N, S, E, W

#define ROWS 1000 // global Height
#define COLUMNS 1000 // global Width

int offset2D(int row, int col); // does the math for pointer access
float getVal(float* ary, int row, int col); // returns the value of ary[row][col]
void setVal(float* ary, int row, int col, float val); // sets the value of ary[row][col]
void swapArays(float** ary1, float** ary2); // Swap 2 arrays by pointers
int* calculatePixel(float temp); // calculates the rgb values for a given temperature
void printPPM(FILE* file, float* plate, int params[]); // prints the plate to a ppm image
char* createFileName(char* fileName, int numThreads, int numProcess); // create the file name for the ppm


/* values to pass in: (i.e. spots in argv[])
	0. name of program
	1. epsilon
	2. max iteration count
	3. number of threads for openMP
*/
int main(int argc, char* argv[])
{
	float	epsilon;		// epsilon value to compare gDiffNorm with
	int		maxIter;		// maximum number of iterations
	double	startT, stopT;	// wallclock timer
	double	time;			// time to runProgram
	int		r, c;			// rows and columns
	int		iterCnt;		// number of iterations completed
	int		rFirst, rLast;	// first and last rows for an MPI region
	float	diffNorm;		// local difference between old cells and new cells 
	float	gDiffNorm;		// global difference between old cells and new cells
	float*	oldPlate; 		// arrays to hold the old and new cells
	float*	newPlate;
	int size;				// size of local cell arrays
	int nThreads;			// size of thread team
	int commSize;			// commSize for MPI
	int myRank;				// who am I for MPI
	MPI_Status	status;		// MPI Status



	/*################ Initialize ################*/
	// set up MPI region
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &commSize);

	//start stopwatch
	if (myRank == 0)
	{
		startT = omp_get_wtime();
	}

	// get input from commandline
	epsilon = (float)atof(argv[1]);
	maxIter = atoi(argv[2]);
	nThreads = atoi(argv[3]);

	// request nThreads from openMP
	omp_set_num_threads(nThreads);

	// intialise iterations count
	iterCnt = 0;

#if DEBUG
	// print headers
	if (myRank == 0)
	{
		printf("Isaac Lehman ~ COMP322 ~ Laplace Heat Distribution\n");
		printf("epsilon: %f\nmaxIter: %d\n\n", epsilon, maxIter);
	}
#endif


	/*################ Make The Arrays ################*/

	/* SPMD allocation */
	rFirst = (myRank * ROWS) / commSize;		// start spot out of total ROWS
	rLast = ((myRank + 1) * ROWS) / commSize;	// stop spot out of total ROWS
	size = rLast - rFirst + 2;					// the size of array with +2 buffer

	// both 0 indexed so use '<=' in for loops
	rFirst = 1; // starting spot
	rLast = size - 3; // stopping spot

	// make room for boarders
	if (myRank == 0)			rFirst++;
	if (myRank == commSize - 1) rLast--;



	// get memory for cells
	oldPlate = (float*)malloc(size * COLUMNS * sizeof(float));
	newPlate = (float*)malloc(size * COLUMNS * sizeof(float));



	/*################ FIll The Arrays ################*/

	/* Fill the interior cells as the average of the 4 edges */
	float interiorVal = (NORTH + SOUTH + EAST + WEST) / 4.0f;
	for (r = rFirst; r <= rLast; r++)
		for (c = 1; c < COLUMNS - 1; c++) {
			setVal(oldPlate, r, c, interiorVal);
			setVal(newPlate, r, c, interiorVal);
		}

	/* Fill the top and bottom cells as North and South */
	for (c = 0; c < COLUMNS; c++) { 
		setVal(oldPlate, rFirst - 1, c, NORTH);
		setVal(oldPlate, rLast + 1, c, SOUTH);

		setVal(newPlate, rFirst - 1, c, NORTH);
		setVal(newPlate, rLast + 1, c, SOUTH);
	}

	/* Fill the left and right cells as East and West */
	for (r = rFirst; r <= rLast; r++) { // keep top and bottom corners at North and South
		setVal(oldPlate, r, 0, EAST);
		setVal(oldPlate, r, COLUMNS - 1, WEST);

		setVal(newPlate, r, 0, EAST);
		setVal(newPlate, r, COLUMNS - 1, WEST);
	}


	/*################ Loop Until Plate Meets Specs ################*/
	do {

		/* Send up unless I'm at the top, then receive from below */
		if (myRank < commSize - 1)
			MPI_Send((oldPlate + rLast * COLUMNS), COLUMNS, MPI_FLOAT, myRank + 1, 0, MPI_COMM_WORLD); // send row rLast
		if (myRank > 0)
			MPI_Recv(oldPlate, COLUMNS, MPI_FLOAT, myRank - 1, 0, MPI_COMM_WORLD, &status); // recieve at row (rFirst - 1) = 0

		/* Send down unless I'm at the bottom */
		if (myRank > 0)
			MPI_Send((oldPlate + rFirst * COLUMNS), COLUMNS, MPI_FLOAT, myRank - 1, 1, MPI_COMM_WORLD); // send row rFirst
		if (myRank < commSize - 1)
			MPI_Recv((oldPlate + (rLast + 1) * COLUMNS), COLUMNS, MPI_FLOAT, myRank + 1, 1, MPI_COMM_WORLD, &status); // recieve at row rLast + 1

		/* Compute new values (but not on boundary) */
		iterCnt++;
		diffNorm = 0.0f;

#pragma omp parallel for reduction(+:diffNorm)
		for (r = rFirst; r <= rLast; r++) {

			if (myRank == 0 && r == rFirst && iterCnt == 1) // print out how many threads and processors we got
			{
				printf("\nWe got %d/%d Threads and %d processors\n\n", omp_get_num_threads(), nThreads, commSize);
			}

			int cp;// local loop variable for parallel region
			for (cp = 1; cp < COLUMNS - 1; cp++) {
				//compute “new” values: avg neighbors North-South-East-West
				float newVal = (getVal(oldPlate, r, cp + 1) + getVal(oldPlate, r, cp - 1) +
					getVal(oldPlate, r + 1, cp) + getVal(oldPlate, r - 1, cp)) / 4.0f;
				setVal(newPlate, r, cp, newVal);

				// calculate difference between old cell and new cell
				diffNorm += (getVal(newPlate, r, cp) - getVal(oldPlate, r, cp)) *
					(getVal(newPlate, r, cp) - getVal(oldPlate, r, cp));
			}
		}

		// transfer points - swap arrays
		swapArays(&oldPlate, &newPlate);

		// combine all the diffnorms into a global diffnorm
		MPI_Allreduce(&diffNorm, &gDiffNorm, 1, MPI_FLOAT, MPI_SUM,
			MPI_COMM_WORLD);
		gDiffNorm = sqrt(gDiffNorm);

#if DEBUG
		if (myRank == 0 && iterCnt % 1000 == 0) { // if master and every 1000 iterations, send the user a progress update
			stopT = omp_get_wtime();	//stop stopwatch
			time = stopT - startT;
			printf("At iteration %d, diff: %.4f, elapsed time: %.4f seconds\n", iterCnt, gDiffNorm, time);
		}
#endif

	} while (gDiffNorm > epsilon && iterCnt < maxIter);

	/*################ Print PPM Picture ################*/

	// send all plates back to master
	if (myRank != 0) {
		int sendParams[3] = { size, rFirst, rLast }; // params to send to master
		MPI_Send(sendParams, 3, MPI_INT, 0, 2, MPI_COMM_WORLD); // send params, tag 2
		MPI_Send(oldPlate, size * COLUMNS, MPI_FLOAT, 0, 3, MPI_COMM_WORLD); // send plate, tag 3
	}

	// recieve in order and print to ppm
	if (myRank == 0) 
	{
		int recvParams[3] = { size, rFirst, rLast }; // params to recieve from other nodes
		char* fileName = createFileName("heatTransfer", nThreads, commSize);

		FILE* ppmFile = fopen(fileName, "w"); // ppm file that will store the picture
		fprintf(ppmFile, "P3\n%d %d #image width (cols) & height (rows)\n", COLUMNS, ROWS); // header
		fprintf(ppmFile, "# Isaac Lehman ~ COMP322 ~ Laplace Heat Distribution\n");
		fprintf(ppmFile, "# Run with %d processers and %d threads\n", commSize, nThreads);
		fprintf(ppmFile, "# This image took %d iterations to converge in %.4f seconds\n", iterCnt, time);
		fprintf(ppmFile, "255		#maximum pixel value\n"); // max rgb value
		
		// write masters plate to file
		printPPM(ppmFile, oldPlate, recvParams);

		// write other nodes to plate in order
		int process;
		for (process = 1; process < commSize; process++)
		{
			MPI_Recv(recvParams, 3, MPI_INT, process, 2, MPI_COMM_WORLD, &status); // recieve parameters

			// if the arrays was a different size to due spmd allocation, make it the right size
			if (size < recvParams[0])
			{
				size = recvParams[0];
				free(oldPlate);
				oldPlate = NULL;
				oldPlate = (float*)malloc(recvParams[0] * COLUMNS * sizeof(float));
			}

			MPI_Recv(oldPlate, recvParams[0] * COLUMNS, MPI_FLOAT, process, 3, MPI_COMM_WORLD, &status); 

			// print the nodes plate to the file 
			printPPM(ppmFile, oldPlate, recvParams);
		}
		fclose(ppmFile); // close the file
	}
	

	// memory cleanup
	free(newPlate);
	newPlate = NULL;
	free(oldPlate);
	oldPlate = NULL;

	// end MPI region
	MPI_Finalize();

	// print final stats
	if (myRank == 0)
	{
		stopT = omp_get_wtime();	//stop stopwatch
		time = stopT - startT;
		printf("Plate calculation completed, Time: %.4f seconds, Final iteration %d, Final diff is %.4f\n", time, iterCnt, diffNorm);

		printf("\n\n\t<< Normal Termination >> \n");
	}
	return 0;
}


/*
	does the math for pointer access
*/
int offset2D(int row, int col) {
	return (row * COLUMNS) + col;
}

/*
	returns the value of ary[row][col]
*/
float getVal(float* ary, int row, int col) {
	return *(ary + offset2D(row, col));
}

/*
	sets the value of ary[row][col]
*/
void setVal(float* ary, int row, int col, float val) {
	*(ary + offset2D(row, col)) = val;
}

/*
	Swap 2 arrays by pointers
*/
void swapArays(float** ary1, float** ary2) {
	float* temp = *ary1;
	*ary1 = *ary2;
	*ary2 = temp;
}

/*
	pixels at 100 = pure red (255, 0, 0)
	pixels at 0   = pure blue (0, 0, 255)

	To interpolate:
		Red: (Temp/TH) * 255)
		Blue: (1.0 - Temp/TH) * 255

		in our case TH = 100, the high temperature

		green = 0 always.
*/
int* calculatePixel(float temp) {
	int pixel[3] = { 0 }; // initialise to 0

	pixel[0] = (int)((temp / MAX_TEMP) * 255); // RED
	pixel[2] = (int)((1 - temp / MAX_TEMP) * 255); // BLUE

	return pixel;
}

/*
	create full file name (filename + _numThread + T + _numProcess + P + .ppm)
*/
char* createFileName(char* fileName, int numThreads, int numProcess) {
	// get the number of processes and threads as strings
	char numT[12];
	sprintf(numT, "%d", numThreads);
	char numP[12];
	sprintf(numP, "%d", numProcess);

	char* fileType = ".ppm"; // type of file 

	// get a memory spot big enough to hold new file name 
	// (+4) for "_T_P"
	// (+1) for the null charachter
	int buffer = strlen(fileName) + strlen(numT) + strlen(numP) + strlen(fileType) + 4 + 1;
	char* finalFileName = (char*)malloc(buffer);

	// put it all together
	strcat(finalFileName, fileName);
	strcat(finalFileName, "_");
	strcat(finalFileName, numT);
	strcat(finalFileName, "T");
	strcat(finalFileName, "_");
	strcat(finalFileName, numP);
	strcat(finalFileName, "P");
	strcat(finalFileName, fileType);
	return finalFileName;
}


/*
	Print the plate to a ppm file colorazed blue to red (cold to hot)
*/
void printPPM(FILE* file, float* plate, int params[]) {
	// write 5 rgb value sets to each line in the ppm file
	int r;
	for (r = params[1]; r <= params[2]; r++) {
		int c = 0;
		while (c < COLUMNS) {
			int z = 0;
			while (z < 5 && c < COLUMNS) { // print 5 pixels per one line
				// get the pixel
				int* pixel = calculatePixel(getVal(plate, r, c));
				int r = pixel[0];
				int g = pixel[1];
				int b = pixel[2];

				// append to string
				fprintf(file, "%d %d %d ", r, g, b);

				z++;
				c++;
			}
			// write line to file
			fprintf(file, "\n");
		}
	}
}