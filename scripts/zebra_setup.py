import json
import urllib.request
import os
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

# first we load the data from LILA, and create a Python loop to retrieve a subset of data
# process pulled from here: https://github.com/microsoft/CameraTraps/blob/master/data_management/download_lila_subset.py
sas_url = 'https://lilablobssc.blob.core.windows.net/snapshot-safari/KRU/KRU_public?st=2020-01-01T00%3A00%3A00Z&se=2034-01-01T00%3A00%3A00Z&sp=rl&sv=2019-07-07&sr=c&sig=0qPMsAsGwKMGLGuPfoNKzDa5Agi5QEC73wLNkMEY0KE%3D'
json_file = r'../SnapshotKruger_S1_v1.0.json'
output_dir = r'../data_sets/'

# use_azcopy_for_download = False
overwrite_files = False
n_download_threads = 25

base_url = sas_url.split('?')[0]
sas_token = sas_url.split('?')[1]
os.makedirs(output_dir, exist_ok=True)


# now we open the data store file
def pull_data(json_filename):
    with open(json_filename, 'r') as f:
        data = json.load(f)

    categories = data['categories']
    annotations = data['annotations']
    images = data['images']

    # get all files except for those whose categories we want to omit
    categories_of_interest = list(categories)
    category_ids = []
    for category in categories_of_interest:
        category_ids.append(category['id'])

    image_ids = set([ann['image_id'] for ann in annotations if ann['category_id'] in category_ids])

    print('Selected {} of {} images'.format(len(image_ids), len(images)))

    filenames = [im['file_name'] for im in images if im['id'] in image_ids]
    assert len(filenames) == len(image_ids)

    return filenames


def download_image(fn):
    url = base_url + '/' + fn
    print('processing')
    target_file = os.path.join(output_dir, fn)
    if (not overwrite_files) and (os.path.isfile(target_file)):
        print('Skipping file {}'.format(fn))
    else:
        print('Downloading {} to {}'.format(url, target_file))
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        urllib.request.urlretrieve(url, target_file)


# Download the image files
'''if use_azcopy_for_download:

    print('Downloading images for {0} with azcopy'.format(species_of_interest))

    # Write out a list of files, and use the azcopy "list-of-files" option to download those files
    # this azcopy feature is unofficially documented at https://github.com/Azure/azure-storage-azcopy/wiki/Listing-specific-files-to-transfer
    az_filename = os.path.join(output_dir, 'filenames.txt')
    with open(az_filename, 'w') as f:
        for fn in filenames:
            f.write(fn.replace('\\', '/') + '\n')
    cmd = 'azcopy cp "{0}" "{1}" --list-of-files "{2}"'.format(
        sas_url, output_dir, az_filename)
    os.system(cmd)'''

filenames = pull_data(json_file)

# Loop over files
print('Downloading images for {0} without azcopy'.format("multiple species"))

if n_download_threads <= 1:
    for fn in tqdm(filenames):
        download_image(fn)

else:
    pool = ThreadPool(n_download_threads)
    tqdm(pool.imap(download_image, filenames), total=len(filenames))

print('Done!')
