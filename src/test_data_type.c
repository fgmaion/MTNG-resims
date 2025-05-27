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

#define FILE_INDEX "/cosmos_storage/simulations/TNG_Family/MTNG/groups_264/subhalo_treelink_264.0.hdf5"

int main(void)
{

    hid_t  file_id, dataset_id, group_id, attr_id; /* identifiers */
    herr_t status;

    file_id = H5Fopen(FILE_INDEX, H5F_ACC_RDONLY, H5P_DEFAULT);
   
    dataset_id = H5Dopen(file_id, "/Subhalo/TreeID", H5P_DEFAULT);
    hid_t type_id = H5Dget_type(dataset_id);
    print_datatype_details(type_id);
    
    //status = H5Dread(dataset_id, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, offset);
    //status = H5Dclose(dataset_id);
   
    /* Close the file. */
    status = H5Fclose(file_id);

} 
