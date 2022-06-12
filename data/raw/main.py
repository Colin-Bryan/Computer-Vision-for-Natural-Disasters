import os
import tarfile
import zipfile

# Please make sure to download the files named "AIDER_filtered.zip" and "data_disaster_types.tar.gz"
#   and add them to this folder before extracting

def main():
    aider_prefiltered_filename = "AIDER_filtered.zip"
    # Get path to "AIDER_filtered.zip" to use for extracting
    path_aider_prefiltered = os.path.join(os.getcwd(), aider_prefiltered_filename)
    print(f"Extracting AIDER_filtered.zip to {path_aider_prefiltered}")
    # Return a ZipFile object for the pathname path_aider_prefiltered
    zip_ref = zipfile.ZipFile(path_aider_prefiltered, 'r')
    zip_ref.extractall(os.getcwd()) # extract
    zip_ref.close()
    print(f"Done extracting {aider_prefiltered_filename}")

    medic_original_filename = "data_disaster_types.tar.gz"
    # Get path to "data_disaster_types.tar.gz" to use for extracting
    path_medic_original = os.path.join(os.getcwd(), medic_original_filename)
    # Return a TarFile object for the pathname path_medic_original
    tar = tarfile.open(path_medic_original)
    print(f"Extracting data_disaster_types.tar.gz to {path_medic_original}")
    tar.extractall(os.getcwd()) # extract
    tar.close()
    print(f"Done extracting {medic_original_filename}")

if __name__ == "__main__":
    main()
