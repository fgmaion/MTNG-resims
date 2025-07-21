#include "hdf5.h"

void read_int_attr(hid_t group_id, const char* attr_name, int* out)
{
    hid_t attr_id;
    herr_t status;
    
    attr_id = H5Aopen(group_id, attr_name, H5P_DEFAULT);
    status = H5Aread(attr_id, H5T_NATIVE_INT, out);
    
    /* Close the attribute. */
    status = H5Aclose(attr_id);

}

void print_datatype_details(hid_t type_id) {
    H5T_class_t type_class = H5Tget_class(type_id);
    size_t type_size = H5Tget_size(type_id);  // Size in bytes

    if (type_class == H5T_INTEGER) {
        H5T_sign_t sign = H5Tget_sign(type_id);
        printf("Dataset type: Integer (%s)\n", (sign == H5T_SGN_NONE) ? "Unsigned" : "Signed");
        printf("Size: %zu bytes (%zu bits)\n", type_size, type_size * 8);
    } 
    else if (type_class == H5T_FLOAT) {
        printf("Dataset type: Floating-point\n");
        printf("Size: %zu bytes (%zu bits)\n", type_size, type_size * 8);

        size_t precision = H5Tget_precision(type_id);
        printf("Precision: %zu bits\n", precision);
    } 
    else if (type_class == H5T_STRING) {
        printf("Dataset type: String\n");
        printf("Size: %zu bytes (variable length if >1 byte per char)\n", type_size);
    } 
    else if (type_class == H5T_COMPOUND) {
        printf("Dataset type: Compound (struct-like)\n");
    } 
    else {
        printf("Dataset type: Unknown or unsupported\n");
    }
}

#define FILE_OFF "/cosmos_storage/home/fgmaion/prob-bias/MTNG/tree_data/offsets.bin"
#define FILE_MP "/cosmos_storage/home/fgmaion/prob-bias/MTNG/tree_data/main_progs.bin"
#define FILE_FP "/cosmos_storage/home/fgmaion/prob-bias/MTNG/tree_data/first_progs.bin"
#define FILE_NP "/cosmos_storage/home/fgmaion/prob-bias/MTNG/tree_data/next_progs.bin"
#define FILE_LT "/cosmos_storage/home/fgmaion/prob-bias/MTNG/tree_data/len_type.bin"

int main(void)
{
    int tree_max = 6885892;

    hid_t  file_id, dataset_id, group_id, attr_id; /* identifiers */
    herr_t status;
    int cum_Ntrees, Ntrees[1], Nhalos[1];
    int i=0;
    long int Ntree_total = 0;

    const char* base = "/cosmos_storage/simulations/TNG_Family/MTNG/treedata/trees.%d.hdf5";
    char file[1000];
 
//    FILE *file_off = fopen(FILE_OFF, "wb");
//    FILE *file_mp = fopen(FILE_MP, "wb");
//    FILE *file_fp = fopen(FILE_FP, "wb");
//    FILE *file_np = fopen(FILE_NP, "wb");
    FILE *file_lt = fopen(FILE_LT, "wb");

    while(Ntree_total < tree_max)
    {
        sprintf(file, base, i);
        /* Open an existing file. */
        file_id = H5Fopen(file, H5F_ACC_RDONLY, H5P_DEFAULT);
    
        /* Open an existing group. */
        group_id = H5Gopen(file_id, "/Header", H5P_DEFAULT);
    
        /* Read basic information on the stored data */
        read_int_attr(group_id, "Ntrees_ThisFile", Ntrees);
        read_int_attr(group_id, "Nhalos_ThisFile", Nhalos);
        
        Ntree_total += Ntrees[0];
        
        printf("Now reading file %d \n", i);
        printf("Ntrees =  %d \n", Ntrees[0]);
        printf("Nhalos =  %d \n", Nhalos[0]);
 
        /* Close the group. */
        status = H5Gclose(group_id); 
   
        // ['LastSnapShotNr', 'Nhalos_ThisFile', 'Nhalos_Total', 'Ntrees_ThisFile', 'Ntrees_Total', 'NumFiles']>

//        // Read the tree offsets
//        long long offset[Ntrees[0]];
//        dataset_id = H5Dopen(file_id, "/TreeTable/StartOffset", H5P_DEFAULT);
//        status = H5Dread(dataset_id, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, offset);
//        status = H5Dclose(dataset_id);
//
//        fwrite(offset, sizeof(long long), Ntrees[0], file_off);
//
//        // Read the Main Progenitors
//        long long main_prog[Nhalos[0]];
//        dataset_id = H5Dopen(file_id, "/TreeHalos/TreeMainProgenitor", H5P_DEFAULT);
//        status = H5Dread(dataset_id, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, main_prog);
//        //hid_t type_id = H5Dget_type(dataset_id);
//        //print_datatype_details(type_id);
//        status = H5Dclose(dataset_id);
//        fwrite(main_prog, sizeof(long long), Nhalos[0], file_mp);
//
//        // Read the First Progenitors
//        long long first_prog[Nhalos[0]];
//        dataset_id = H5Dopen(file_id, "/TreeHalos/TreeFirstProgenitor", H5P_DEFAULT);
//        status = H5Dread(dataset_id, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, first_prog);
//        //hid_t type_id = H5Dget_type(dataset_id);
//        //print_datatype_details(type_id);
//        status = H5Dclose(dataset_id);
//        fwrite(first_prog, sizeof(long long), Nhalos[0], file_fp);
//
//        // Read the Next Progenitors
//        long long next_prog[Nhalos[0]];
//        dataset_id = H5Dopen(file_id, "/TreeHalos/TreeNextProgenitor", H5P_DEFAULT);
//        status = H5Dread(dataset_id, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, next_prog);
//        //hid_t type_id = H5Dget_type(dataset_id);
//        //print_datatype_details(type_id);
//        status = H5Dclose(dataset_id); 
//        fwrite(next_prog, sizeof(long long), Nhalos[0], file_np);
 
        // Read the Stellar Mass
        int lentype[6*Nhalos[0]];
        int star_len[Nhalos[0]];
        dataset_id = H5Dopen(file_id, "/TreeHalos/SubhaloLenType", H5P_DEFAULT);
        status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, lentype);
        hid_t type_id = H5Dget_type(dataset_id);
        print_datatype_details(type_id);
        status = H5Dclose(dataset_id); 
        for(int j=0; j<Nhalos[0]; j++)
        {
            star_len[j] = lentype[4+6*j];
        }

        fwrite(star_len, sizeof(int), Nhalos[0], file_lt);
    
//        printf("File offsets are %lld and %lld \n", offset[0], offset[1]);
//        printf("First few progenitors are %ld, %ld and %ld \n", main_prog[0], main_prog[1], main_prog[2]);
//        printf("\n \n");

        /* Close the file. */
        status = H5Fclose(file_id);

        i++;
   }

//   fclose(file_off);
//   fclose(file_mp);
//   fclose(file_fp);
//   fclose(file_np);
    fclose(file_lt);
}

