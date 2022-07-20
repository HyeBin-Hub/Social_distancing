import shutil
import wget

### 1
filename= "//social_distancing_d.zip"
extract_dir= "//Social_distancing/data"
archive_format="zip"

# shutil.unpack_archive(filename,extract_dir,archive_format)

### 2
filename= "/social-distance.zip"
extract_dir= "//Social_distancing/data"
archive_format="zip"

# shutil.unpack_archive(filename,extract_dir,archive_format)


model_url="https://pjreddie.com/media/files/yolov3.weights"
config_url="https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"

out_dir= "//Social_distancing/pretrained"

# wget.download(model_url,out_dir)
# wget.download(config_url,out_dir)