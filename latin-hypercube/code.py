import numpy as np

# Function to read the .txt file
def read_config_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

# Function to modify the values
def modify_values(lines, modifications):
    modified_lines = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            param_name = parts[0]
            if param_name in modifications:
                if len(modifications[param_name])>1:
                    new_value1 = modifications[param_name][0]
                    new_value2 = modifications[param_name][1]

                    # Replace the old value with the new one
                    parts[1] = str(np.round(new_value1, decimals=5))
                    parts[2] = str(np.round(new_value2, decimals=5))
                else:
                    new_value = modifications[param_name][0]
                    # Replace the old value with the new one
                    parts[1] = str(np.round(new_value, decimals=5))
                    
                line = '    '.join(parts) + '\n'
        modified_lines.append(line)
    return modified_lines

# Function to save the modified lines to a new file
def save_config_file(file_path, lines):
    with open(file_path, 'w') as file:
        file.writelines(lines)

# Main script
def main():
    input_file = 'param_MTNG-hydro.txt'   # Path to the input file

    import deepdish as dd
    npoints = 30
    seed = 1997
    lh_data = dd.io.load("/cosmos_storage/data_sharing/MN5_resims/cpars_{0}_{1}.h5".format(npoints, seed))
    
    for i in range(lh_data['cpars'].shape[0]):
    
        output_file = 'param_MTNG-hydro_{:d}.txt'.format(i) # Path to save the modified file
        
        # Modifications you want to make
        modifications = {
            'WindEnergyIn1e51erg': [lh_data['cpars'][i,0]],
            'VariableWindVelFactor': [lh_data['cpars'][i,1]],
            'MaxSfrTimescale': [1e-3*lh_data['cpars'][i,2], lh_data['cpars'][i,2]],
            'WindFreeTravelDensFac': [lh_data['cpars'][i,3]],
            'RadioFeedbackFactor': [lh_data['cpars'][i,4]],
            'BlackHoleFeedbackFactor': [lh_data['cpars'][i,5]],
            'RadioFeedbackReiorientationFactor': [lh_data['cpars'][i,6]]
        }
    
        # Read the file
        lines = read_config_file(input_file)
    
        # Modify the values
        modified_lines = modify_values(lines, modifications)
    
        # Save the new file
        save_config_file(output_file, modified_lines)
    
        print(f"File saved as {output_file}")

if __name__ == "__main__":
    main()

