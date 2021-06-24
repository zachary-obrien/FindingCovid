import os
from PIL import Image
import sys
from pydicom import dcmread
from shutil import copyfile


def convert_dcm_dir(folder_name):
    image_count = 5000
    for path, subdirs, files in os.walk(folder_name):
        png_files = []
        for filename in files:
            filename = path + "/" + filename
            if filename.endswith(".dcm"):
                if image_count == 0:
                    exit()
                try:
                    base_file = os.path.basename(filename)
                    file_path = '/media/zac/12TB Drive/covid-detector/extracted_images/pngs/'
                    img_filename = file_path + base_file.replace(".dcm", ".png")

                    if not os.path.isfile(img_filename):
                        print("Writing file", image_count, ":", img_filename)
                        image_count = image_count - 1
                        ds1 = dcmread(filename)
                        # if image_count > 495:
                        #     print(ds1.file_meta)
                        img_to_save = ds1.pixel_array

                        array_buffer = img_to_save.tobytes()
                        img = Image.new("I", img_to_save.T.shape)
                        img.frombytes(array_buffer, 'raw', "I;16")
                        img.save(img_filename)
                        png_files.append(img_filename)
                    # else:
                        # print("File already exists:", img_filename)

                except ValueError:
                    try:
                        print("Value Error on", filename, "but trying to continue:", sys.exc_info()[0])
                        img = Image.fromarray(img_to_save)
                        img.save(img_filename)
                        png_files.append(img_filename)
                        print("Writing file", image_count, ":", img_filename)
                    except:
                        print("Still couldn't open/process dcm:", sys.exc_info()[0])
                        raise
                except:
                    print("Couldn't open/process dcm:", sys.exc_info()[0])
                    raise
            elif filename.endswith(".csv"):
                print("is csv:", filename)
                new_folder = '/media/zac/12TB Drive/covid-detector/extracted_images/'
                new_filename = new_folder + os.path.basename(filename)
                print("New location", new_filename)
                copyfile(filename, new_filename)

convert_dcm_dir("/media/zac/12TB Drive/covid-detector/unpacked_data")