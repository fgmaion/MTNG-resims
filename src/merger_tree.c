#include <stdio.h>
#define NELEMS(x)  (sizeof(x) / sizeof((x)[0]))

// Function to get all progenitors for a given halo
void get_all_progs(int *roots, int snap_0, int depth,
                   int *firstprog, int *nextprog) {

    size_t num_roots = NELEMS(roots);
    
    // PUT THE ROOTS INSIDE THE FIRST FILE
    char name[100];
    sprintf(name, "all_progs_%d.txt", snap_0);

    // open the file for output
    FILE *file = fopen(name, "w");
    if (file == NULL) {
        printf("Error opening file.\n");
        return;
    }

    // Fill in the first level of progenitors at snap_0
    for (int ii = 0; ii < num_roots; ii++) {
    
        fprintf(file, "%d\n", roots[ii]);
    }
    fclose(file);
    //////////////////////////////////////////////////

//     for (int i = 0; i < depth; i++) {
//         int snap = snap_0 - (i + 1);

//         sprintf(name, "new_roots.txt", snap);

//         // open the file for output
//         FILE *file = fopen(name, "w");
//         if (file == NULL) {
//             printf("Error opening file.\n");
//             return;
//         }

//         for (int j = 0; j < n; j++) {
//             int root = roots[j];
//             int fp = firstprog[root];

//             while (fp != -1) {
//                 all_idx[pos] = fp;
//                 new_roots[new_num_roots++] = fp;
//                 fp = nextprog[ii * num_halos + fp];
//             }
//         }

//         // Update the list of roots for the next iteration
//         num_roots = new_num_roots;
//         for (int k = 0; k < new_num_roots; k++) {
//             roots[k] = new_roots[k];
//         }

//         // Early termination if no more roots are found
//         if (num_roots == 0) {
//             break;
//         }
//     }
// }
}
