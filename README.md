# Jacobi-Iteration-Heat-Transfer
Models a the heat transfer of a sheet and visualizes it through a PPM image.  Program runs in both shared and distributed memory using both openMP and MPI.



From file:
-------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------
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
