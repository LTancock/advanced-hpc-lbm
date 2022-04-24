/* PASS ROWS OR START + END INSTEAD OF RANK + SIZE SO HALO EXCHANGE IS EASIER
   DO HALO EXCHANGE
   DO COLLATION
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <mpi.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, int rank, int size, int* start, int* rows);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
//int timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, t_speed*firstline, int* obstacles);
int timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, int rank, int size, int start, int rows);
//int compute_cells(const t_param params, t_speed* cells, t_speed* tmp_cells, t_speed* firstline,
                  //int* obstacles);
int compute_cells(const t_param params, t_speed* cells, t_speed* tmp_cells,
                  int* obstacles, int rank, int size, int start, int rows);
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles, int rank, int size, int start, int rows);
int propagate(const t_param params, t_speed* cells, t_speed* tmp_cells);
int rebound(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, int rank);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* cells, int* obstacles, int rank, int size, int start, int rows);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  int size, rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  //printf("rank %d\n", rank);
  //printf("size %d\n", size);

  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;                                                             /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* Total/init time starts here: initialise our data structures and load values from file */
  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic=tot_tic;
  //initialise in each rank instead, splitting up initialisation

  //maybe make this return the start and rows instead so less computation needs to happen
  int start, rows;
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels, rank, size, &start, &rows);
  printf("main start = %d, rows = %d, rank = %d\n", start, rows, rank);
  /*int* sendcounts = malloc(sizeof(int) * size);
  for (int r = 0; r < size; r++){
    sendcounts[r] = (params.nx*params.ny)/size + ((params.ny % size) > r) ? params.nx:0;//no. of cells for each rank
    
  }
  MPI_Scatterv(&cells, sendcounts, starting line?, t_speed, sendcounts[rank]?, t_speed, 0, ??) no. of lines (cells)
  MPI_Scatterv(&tmp_cells, sendcounts, starting line?, t_speed, sendcounts[rank]?, t_speed, 0, ??) no. of lines (tmp_cells)
  sendrec obstacles, av_vels
  */
  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    //if (rank == 3) printf("tt = %d, rank = %d\n", tt, rank);
    timestep(params, cells, tmp_cells, obstacles, rank, size, start, rows);
    //if (rank == 3) printf("hi1\n");
    t_speed* temp = cells;
    cells = tmp_cells;
    tmp_cells = temp;
    //if (rank == 3) printf("hi2\n");
    av_vels[tt] = av_velocity(params, cells, obstacles, rank, size, start, rows);
    //if (rank == 3) printf("hi3\n");
  
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }
  printf("alltimestep rank = %d\n", rank);
  
  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;

  // Collate data from ranks here 
  //DO THIS
  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;
  
  /* write final values and free memory */
  printf("Hello from rank %d of %d\n", rank, size);
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
  printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
  printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
  printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
  printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels, rank);

  MPI_Finalize();

  return EXIT_SUCCESS;
}
//int timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, t_speed* firstline, int* obstacles)
int timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, int rank, int size, int start, int rows)
{
  //printf("hia rank = %d\n", rank);
  if (rank == size-1) accelerate_flow(params, cells, obstacles, rank, size, start, rows);
  //if (rank == 3) printf("hi0.1 rank = %d\n", rank);
  compute_cells(params, cells, tmp_cells, obstacles, rank, size, start, rows);
  //if (rank == 3) printf("hi0.2 rank = %d\n", rank);
  //printf("compute no errors rank = %d\n", rank);
  /*
  propagate(params, cells, tmp_cells);
  rebound(params, cells, tmp_cells, obstacles);
  collision(params, cells, tmp_cells, obstacles);
  */

  return EXIT_SUCCESS;
}
//int compute_cells(const t_param params, t_speed* cells, t_speed* tmp_cells, t_speed* firstline,
                  //int* obstacles)
int compute_cells(const t_param params, t_speed* cells, t_speed* tmp_cells, 
                  int* obstacles, int rank, int size, int start, int rows)
{

  //t_speed* cells = *cells_ptr;
  //t_speed* tmp_cells = *tmp_cells_ptr;
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  const float inv_c_sq = 3.f;
  const float inv_c_sq2 = 4.5f;

  /* loop over _all_ cells *///wrong
  //doesn't account for extra row on some atm
  for (int jj = 0; jj < rows; jj++)
  {
    //printf("jj = %d, rank = %d\n", jj, rank);
    int y_n = (jj + 1) % rows;
    int y_s = (jj == 0) ? (jj + rows - 1) : (jj - 1);
    for (int ii = 0; ii < params.nx; ii++)
    {
      //propagate
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
     //take these out of the loop
      //int y_n = (jj + 1) % params.ny;
      int x_e = (ii + 1) % params.nx;
      //int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
      int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      const float speed0 = cells[ii + jj*params.nx].speeds[0]; /* central cell, no movement */
      const float speed1 = cells[x_w + jj*params.nx].speeds[1]; /* east */
      const float speed2 = cells[ii + y_s*params.nx].speeds[2]; /* north */
      const float speed3 = cells[x_e + jj*params.nx].speeds[3]; /* west */
      const float speed4 = cells[ii + y_n*params.nx].speeds[4]; /* south */
      const float speed5 = cells[x_w + y_s*params.nx].speeds[5]; /* north-east */
      const float speed6 = cells[x_e + y_s*params.nx].speeds[6]; /* north-west */
      const float speed7 = cells[x_e + y_n*params.nx].speeds[7]; /* south-west */
      const float speed8 = cells[x_w + y_n*params.nx].speeds[8]; /* south-east */
      //printf("hi1\n");
      //rebound

      /* if the cell contains an obstacle */
      if (obstacles[jj*params.nx + ii])
      {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        /*
        cells[ii + jj*params.nx].speeds[1] = tmp_cells[ii + jj*params.nx].speeds[3];
        cells[ii + jj*params.nx].speeds[2] = tmp_cells[ii + jj*params.nx].speeds[4];
        cells[ii + jj*params.nx].speeds[3] = tmp_cells[ii + jj*params.nx].speeds[1];
        cells[ii + jj*params.nx].speeds[4] = tmp_cells[ii + jj*params.nx].speeds[2];
        cells[ii + jj*params.nx].speeds[5] = tmp_cells[ii + jj*params.nx].speeds[7];
        cells[ii + jj*params.nx].speeds[6] = tmp_cells[ii + jj*params.nx].speeds[8];
        cells[ii + jj*params.nx].speeds[7] = tmp_cells[ii + jj*params.nx].speeds[5];
        cells[ii + jj*params.nx].speeds[8] = tmp_cells[ii + jj*params.nx].speeds[6];
        */
        //const t_speed temp = tmp_cells[ii + jj*params.nx];
        tmp_cells[ii + jj*params.nx].speeds[1] = speed3;
        tmp_cells[ii + jj*params.nx].speeds[2] = speed4;
        tmp_cells[ii + jj*params.nx].speeds[3] = speed1;
        tmp_cells[ii + jj*params.nx].speeds[4] = speed2;
        tmp_cells[ii + jj*params.nx].speeds[5] = speed7;
        tmp_cells[ii + jj*params.nx].speeds[6] = speed8;
        tmp_cells[ii + jj*params.nx].speeds[7] = speed5;
        tmp_cells[ii + jj*params.nx].speeds[8] = speed6;
        
      }
      //printf("hi2\n");
      
      //collision

      /* don't consider occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* compute local density total */
        float local_density = 0.f;

        /*
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += tmp_cells[ii + jj*params.nx].speeds[kk];
        }
        */
        local_density += speed0 + speed1 + speed2 + speed3 + speed4 + speed5 + 
                          speed6 + speed7 + speed8;

        float inv_local_density = 1/local_density;

        /* compute x velocity component */
        float u_x = (speed1
                      + speed5
                      + speed8
                      - (speed3
                         + speed6
                         + speed7))
                     * inv_local_density;
        /* compute y velocity component */
        float u_y = (speed2
                      + speed5
                      + speed6
                      - (speed4
                         + speed7
                         + speed8))
                     * inv_local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        // zero velocity density: weight w0
        /*
        d_equ[0] = w0 * local_density
                   * (1.f - u_sq / (2.f * c_sq));
        // axis speeds: weight w1
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                         + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                         + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                         + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                         + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        // diagonal speeds: weight w2 
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                         + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                         + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                         + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                         + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        */

        /* zero velocity density: weight w0 */
        //having 0.5f instead of just 0.5 makes it significantly faster
        d_equ[0] = w0 * local_density
                   * (1.f - u_sq * 0.5f * inv_c_sq);
        // axis speeds: weight w1
        d_equ[1] = w1 * local_density * (1.f + u[1] * inv_c_sq
                                         + (u[1] * u[1]) * (inv_c_sq2)
                                         - u_sq * (0.5f * inv_c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] * inv_c_sq
                                         + (u[2] * u[2]) * (inv_c_sq2)
                                         - u_sq * (0.5f * inv_c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] * inv_c_sq
                                         + (u[3] * u[3]) * (inv_c_sq2)
                                         - u_sq * (0.5f * inv_c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] * inv_c_sq
                                         + (u[4] * u[4]) * (inv_c_sq2)
                                         - u_sq * (0.5f * inv_c_sq));
        // diagonal speeds: weight w2
        d_equ[5] = w2 * local_density * (1.f + u[5] * inv_c_sq
                                         + (u[5] * u[5]) * (inv_c_sq2)
                                         - u_sq * (0.5f * inv_c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] * inv_c_sq
                                         + (u[6] * u[6]) * (inv_c_sq2)
                                         - u_sq * (0.5f * inv_c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] * inv_c_sq
                                         + (u[7] * u[7]) * (inv_c_sq2)
                                         - u_sq * (0.5f * inv_c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] * inv_c_sq
                                         + (u[8] * u[8]) * (inv_c_sq2)
                                         - u_sq * (0.5f * inv_c_sq));
        

        /* relaxation step */
        /*
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          
          tmp_cells[ii + jj*params.nx].speeds[kk] = tmp_cells[ii + jj*params.nx].speeds[kk]
                                                  + params.omega
                                                  * (d_equ[kk] - tmp_cells[ii + jj*params.nx].speeds[kk]);
        }
        */
        //printf("hi3\n");
        tmp_cells[ii + jj*params.nx].speeds[0] = speed0 + params.omega
                                                  * (d_equ[0] - speed0);
        tmp_cells[ii + jj*params.nx].speeds[1] = speed1 + params.omega
                                                  * (d_equ[1] - speed1);
        tmp_cells[ii + jj*params.nx].speeds[2] = speed2 + params.omega
                                                  * (d_equ[2] - speed2);
        tmp_cells[ii + jj*params.nx].speeds[3] = speed3 + params.omega
                                                  * (d_equ[3] - speed3);
        tmp_cells[ii + jj*params.nx].speeds[4] = speed4 + params.omega
                                                  * (d_equ[4] - speed4);
        tmp_cells[ii + jj*params.nx].speeds[5] = speed5 + params.omega
                                                  * (d_equ[5] - speed5);
        tmp_cells[ii + jj*params.nx].speeds[6] = speed6 + params.omega
                                                  * (d_equ[6] - speed6);
        tmp_cells[ii + jj*params.nx].speeds[7] = speed7 + params.omega
                                                  * (d_equ[7] - speed7);
        tmp_cells[ii + jj*params.nx].speeds[8] = speed8 + params.omega
                                                  * (d_equ[8] - speed8);
      }
    }
  }
  
  //WORKS
  /*
  t_speed* temp = *cells_ptr;
  *cells_ptr = *tmp_cells_ptr;
  *tmp_cells_ptr = temp;
  */
  return EXIT_SUCCESS;
}

//WRONG maybe
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles, int rank, int size, int start, int rows)
{
  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  int jj = rows - 2;

  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj*params.nx]
        && (cells[ii + jj*params.nx].speeds[3] - w1) > 0.f
        && (cells[ii + jj*params.nx].speeds[6] - w2) > 0.f
        && (cells[ii + jj*params.nx].speeds[7] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells[ii + jj*params.nx].speeds[1] += w1;
      cells[ii + jj*params.nx].speeds[5] += w2;
      cells[ii + jj*params.nx].speeds[8] += w2;
      /* decrease 'west-side' densities */
      cells[ii + jj*params.nx].speeds[3] -= w1;
      cells[ii + jj*params.nx].speeds[6] -= w2;
      cells[ii + jj*params.nx].speeds[7] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

int propagate(const t_param params, t_speed* cells, t_speed* tmp_cells)
{
  /* loop over _all_ cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int y_n = (jj + 1) % params.ny;
      int x_e = (ii + 1) % params.nx;
      int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
      int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      tmp_cells[ii + jj*params.nx].speeds[0] = cells[ii + jj*params.nx].speeds[0]; /* central cell, no movement */
      tmp_cells[ii + jj*params.nx].speeds[1] = cells[x_w + jj*params.nx].speeds[1]; /* east */
      tmp_cells[ii + jj*params.nx].speeds[2] = cells[ii + y_s*params.nx].speeds[2]; /* north */
      tmp_cells[ii + jj*params.nx].speeds[3] = cells[x_e + jj*params.nx].speeds[3]; /* west */
      tmp_cells[ii + jj*params.nx].speeds[4] = cells[ii + y_n*params.nx].speeds[4]; /* south */
      tmp_cells[ii + jj*params.nx].speeds[5] = cells[x_w + y_s*params.nx].speeds[5]; /* north-east */
      tmp_cells[ii + jj*params.nx].speeds[6] = cells[x_e + y_s*params.nx].speeds[6]; /* north-west */
      tmp_cells[ii + jj*params.nx].speeds[7] = cells[x_e + y_n*params.nx].speeds[7]; /* south-west */
      tmp_cells[ii + jj*params.nx].speeds[8] = cells[x_w + y_n*params.nx].speeds[8]; /* south-east */
    }
  }

  return EXIT_SUCCESS;
}

int rebound(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
  /* loop over the cells in the grid */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* if the cell contains an obstacle */
      if (obstacles[jj*params.nx + ii])
      {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        cells[ii + jj*params.nx].speeds[1] = tmp_cells[ii + jj*params.nx].speeds[3];
        cells[ii + jj*params.nx].speeds[2] = tmp_cells[ii + jj*params.nx].speeds[4];
        cells[ii + jj*params.nx].speeds[3] = tmp_cells[ii + jj*params.nx].speeds[1];
        cells[ii + jj*params.nx].speeds[4] = tmp_cells[ii + jj*params.nx].speeds[2];
        cells[ii + jj*params.nx].speeds[5] = tmp_cells[ii + jj*params.nx].speeds[7];
        cells[ii + jj*params.nx].speeds[6] = tmp_cells[ii + jj*params.nx].speeds[8];
        cells[ii + jj*params.nx].speeds[7] = tmp_cells[ii + jj*params.nx].speeds[5];
        cells[ii + jj*params.nx].speeds[8] = tmp_cells[ii + jj*params.nx].speeds[6];
      }
    }
  }

  return EXIT_SUCCESS;
}

int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* don't consider occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* compute local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += tmp_cells[ii + jj*params.nx].speeds[kk];
        }

        /* compute x velocity component */
        float u_x = (tmp_cells[ii + jj*params.nx].speeds[1]
                      + tmp_cells[ii + jj*params.nx].speeds[5]
                      + tmp_cells[ii + jj*params.nx].speeds[8]
                      - (tmp_cells[ii + jj*params.nx].speeds[3]
                         + tmp_cells[ii + jj*params.nx].speeds[6]
                         + tmp_cells[ii + jj*params.nx].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (tmp_cells[ii + jj*params.nx].speeds[2]
                      + tmp_cells[ii + jj*params.nx].speeds[5]
                      + tmp_cells[ii + jj*params.nx].speeds[6]
                      - (tmp_cells[ii + jj*params.nx].speeds[4]
                         + tmp_cells[ii + jj*params.nx].speeds[7]
                         + tmp_cells[ii + jj*params.nx].speeds[8]))
                     / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density
                   * (1.f - u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                         + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                         + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                         + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                         + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                         + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                         + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                         + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                         + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));

        /* relaxation step */
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          cells[ii + jj*params.nx].speeds[kk] = tmp_cells[ii + jj*params.nx].speeds[kk]
                                                  + params.omega
                                                  * (d_equ[kk] - tmp_cells[ii + jj*params.nx].speeds[kk]);
        }
      }
    }
  }

  return EXIT_SUCCESS;
}

float av_velocity(const t_param params, t_speed* cells, int* obstacles, int rank, int size, int start, int rows)
{
  //ISSUE communication probably needs to happen, calculate tot_u then divide by tot_cells afterwards?
  //tot_cells = no. of non-blocked cells
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < rows; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        float local_density = 0.f;
        float u_x = 0.f;
        float u_y = 0.f;
        //if not entered since first row blocked off by an obstacle
        /*
        if (jj == 0){
          printf("hi\n");
          u_x = (firstline[ii].speeds[1]
                        + firstline[ii].speeds[5]
                        + firstline[ii].speeds[8]
                        - (firstline[ii].speeds[3]
                           + firstline[ii].speeds[6]
                           + firstline[ii].speeds[7]))
                       / local_density;
          u_y = (firstline[ii].speeds[2]
                        + firstline[ii].speeds[5]
                        + firstline[ii].speeds[6]
                        - (firstline[ii].speeds[4]
                           + firstline[ii].speeds[7]
                           + firstline[ii].speeds[8]))
                       / local_density;
          }

        else{*/
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }
        local_density = 1.f/local_density;
        //compare u_x and u_y with default/changed algorithm
        /* x-component of velocity */
        u_x = (cells[ii + jj*params.nx].speeds[1]
                      + cells[ii + jj*params.nx].speeds[5]
                      + cells[ii + jj*params.nx].speeds[8]
                      - (cells[ii + jj*params.nx].speeds[3]
                         + cells[ii + jj*params.nx].speeds[6]
                         + cells[ii + jj*params.nx].speeds[7]))
                     * local_density;
        /* compute y velocity component */
        u_y = (cells[ii + jj*params.nx].speeds[2]
                      + cells[ii + jj*params.nx].speeds[5]
                      + cells[ii + jj*params.nx].speeds[6]
                      - (cells[ii + jj*params.nx].speeds[4]
                         + cells[ii + jj*params.nx].speeds[7]
                         + cells[ii + jj*params.nx].speeds[8]))
                     * local_density;
        //}
        
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, int rank, int size, int* start, int* rows)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  *rows = ((params->ny % size) > rank) ? 1 + params->ny/size:params->ny/size;//no. of rows for each rank
  *start = ((params->ny % size) > rank) ? rank + rank * (params->ny/size):(params->ny % size) + rank * (params->ny/size);
  //int end = ((params->ny % size) > rank) ? rank + ((rank+1) * params->ny)/size:(params->ny % size) + ((rank+1) * params->ny)/size;
  int end = *start + *rows - 1;
  printf("rank %d, rows = %d, start = %d, end = %d\n", rank, *rows, *start, end);

  //printf("hi1 rank = %d\n", rank);
  /* main grid */
  *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (*rows * params->nx));

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  //printf("hi2 rank = %d\n", rank);
  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (*rows * params->nx));

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  //printf("hi3 rank = %d\n", rank);
  *obstacles_ptr = malloc(sizeof(int) * (*rows * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  //printf("hi4 rank = %d\n", rank);
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj <= *rows; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      (*cells_ptr)[ii + jj*params->nx].speeds[0] = w0;
      /* axis directions */
      (*cells_ptr)[ii + jj*params->nx].speeds[1] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[2] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[3] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[4] = w1;
      /* diagonals */
      (*cells_ptr)[ii + jj*params->nx].speeds[5] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[6] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[7] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[8] = w2;
    }
  }
  //printf("hi5 rank = %d\n", rank);
  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj <= *rows; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      //printf("jj = %d, rank = %d\n", jj, rank);
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }
  //printf("hi6 rank = %d\n", rank);
  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }
  //printf("hi7 rank = %d\n", rank);
  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF/* && yy >= start && yy <= end*/)
  {
    /* some checks */
    //printf("hi7.1 rank = %d\n", rank);
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);
    //printf("hi7.2 rank = %d\n", rank);
    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);
    //printf("hi7.3 rank = %d\n", rank);
    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);
    //printf("hi7.4 rank = %d\n", rank);
    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);
    //printf("yy = %d\n", yy);

    /* assign to array */
    if(yy >= *start && yy <= end){
      (*obstacles_ptr)[xx + yy*params->nx] = blocked;
    }
  }
  //printf("hi8 rank = %d\n", rank);

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);
  //printf("hiinit rank = %d\n", rank);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, int rank)
{
  /*
  ** free up allocated memory
  */
  //printf("hi1 rank = %d\n", rank);
  free(*cells_ptr);
  *cells_ptr = NULL;
  //printf("hi2 rank = %d\n", rank);
  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;
  //printf("hi3 rank = %d\n", rank);
  free(*obstacles_ptr);
  *obstacles_ptr = NULL;
  //printf("hi4 rank = %d\n", rank);
  free(*av_vels_ptr);
  *av_vels_ptr = NULL;
  //printf("hi5 rank = %d\n", rank);
  return EXIT_SUCCESS;
}

//calculate after collate
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles, 0, 1, 0, params.ny) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[ii + jj*params.nx].speeds[kk];
      }
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }

        /* compute x velocity component */
        u_x = (cells[ii + jj*params.nx].speeds[1]
               + cells[ii + jj*params.nx].speeds[5]
               + cells[ii + jj*params.nx].speeds[8]
               - (cells[ii + jj*params.nx].speeds[3]
                  + cells[ii + jj*params.nx].speeds[6]
                  + cells[ii + jj*params.nx].speeds[7]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[ii + jj*params.nx].speeds[2]
               + cells[ii + jj*params.nx].speeds[5]
               + cells[ii + jj*params.nx].speeds[6]
               - (cells[ii + jj*params.nx].speeds[4]
                  + cells[ii + jj*params.nx].speeds[7]
                  + cells[ii + jj*params.nx].speeds[8]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii + params.nx * jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
