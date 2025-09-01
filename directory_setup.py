import os

### SETUP FOLDERS IN "Train_Model" ###
os.makedirs( "Train_Model/TRAINING_IMAGES", exist_ok = True )
os.makedirs( "Train_Model/TRAINING_MASKS", exist_ok = True )

### SETUP FOLDERS IN "Use_Model" ###
os.makedirs( "Use_Model/INPUT_IMAGES", exist_ok = True )
os.makedirs( "Use_Model/OUTPUT_MASKS", exist_ok = True )
os.makedirs( "Use_Model/saved_models", exist_ok = True )

### SETUP FOLDERS IN "utils" ###
os.makedirs( "utils/temp_files/Images", exist_ok = True )
os.makedirs( "utils/temp_files/Masks", exist_ok = True )
os.makedirs( "utils/temp_files/Model_Training/Test_Dataset/Images", exist_ok = True )
os.makedirs( "utils/temp_files/Model_Training/Test_Dataset/Masks", exist_ok = True )
os.makedirs( "utils/temp_files/Model_Training/Training_Dataset/Images", exist_ok = True )
os.makedirs( "utils/temp_files/Model_Training/Training_Dataset/Masks", exist_ok = True )
os.makedirs( "utils/temp_files/Model_Training/Training_Dataset/Training_Images", exist_ok = True )
os.makedirs( "utils/temp_files/Model_Training/Training_Dataset/Training_Masks", exist_ok = True )
os.makedirs( "utils/temp_files/Model_Validation/Generated_Masks", exist_ok = True )
os.makedirs( "utils/temp_files/Model_Validation/temp_crop", exist_ok = True )
os.makedirs( "utils/temp_files/Model_Validation/temp_mask", exist_ok = True )
os.makedirs( "utils/temp_files/output", exist_ok = True )