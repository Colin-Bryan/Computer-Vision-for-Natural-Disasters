import os
import tarfile
import zipfile

# Please make sure to download the files named "AIDER_filtered.zip" and "data_disaster_types.tar.gz"
#   and add them to the data/raw folder before extracting

RAW_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/raw'))

def main():
    # Extracting AIDER_filtered.zip
    aider_prefiltered_input_filename = "AIDER_filtered.zip"
    # Get path to "AIDER_filtered.zip" to use for extracting
    aider_input_path = os.path.join(RAW_DATA_PATH, aider_prefiltered_input_filename)
    print(f"Extracting AIDER_filtered.zip to {RAW_DATA_PATH}")
    # Return a ZipFile object for the pathname aider_input_path
    zip_ref = zipfile.ZipFile(aider_input_path, 'r')
    zip_ref.extractall(RAW_DATA_PATH) # extract to data/raw
    zip_ref.close()
    print(f"Done extracting {aider_prefiltered_input_filename}")

    # Extracting MEDIC dataset (data_disaster_types.tar.gz)
    medic_input_filename = "data_disaster_types.tar.gz"
    # Get path to "data_disaster_types.tar.gz" to use for extracting
    medic_input_path = os.path.join(RAW_DATA_PATH, medic_input_filename)
    # Return a TarFile object for medic_input_path
    tar = tarfile.open(medic_input_path)
    print(f"Extracting data_disaster_types.tar.gz to {RAW_DATA_PATH}")
    tar.extractall(RAW_DATA_PATH) # extract to data/raw
    tar.close()
    print(f"Done extracting {medic_input_filename}")

if __name__ == "__main__":
    main()
