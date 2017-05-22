# import modules
import urllib.request
import sys
from subprocess import call, Popen
import os
import json
import requests

# retrieve file from url
def get_url(url, fname):
    """ Retrieves file from url and returns path to file """
    path_to_file = urllib.request.urlretrieve(url, fname)
    return path_to_file

# extract zip-file url with data
def get_zip_url(url='https://tensorflow.wq.io/sources.json', dataset_id=2):
    # open json file
    j_file = urllib.request.urlopen(url)
    # extract json content
    j_load = json.loads(j_file.read().decode())
    # get id list
    id_list = [x['id'] for x in j_load['list']]
    # get url for zip file
    if (dataset_id < 0):
        # most recent data set
        list_id = id_list.index(id_list[-1])
        zip_url = j_load['list'][list_id]['file']
        actual_id = id_list[-1]
    else:
        list_id = id_list.index(dataset_id)
        zip_url = j_load['list'][list_id]['file']
        actual_id = dataset_id
    return zip_url, actual_id


# upload result zip-file
def upload_zip(res_file, url='https://tensorflow.wq.io/graphs.json',
               ds_id=-1, name="test",
               filename = "filename",
               desc=""):
    fileobj = open(res_file, 'rb')
    r = requests.post(url, data={"source_id": dataset_id,
                                 "name": name,
                                 "description": desc},
                      files={"file": (filename, fileobj)})
    r

# main program
# USAGE: python3 main.py data_set_id describtion_of_result
if __name__ == "__main__":

    # Path Parameters
    path_to_file = "/host/data/data.zip"
    path_to_data = "/host/data/"
    path_to_tf = '/tensorflow/'

    # url for images (input/output)
    ds_id = int(sys.argv[1])
    desc = str(sys.argv[2])
    url_images, dataset_id = get_zip_url(dataset_id=ds_id)
    images_name = url_images.split("/")[-1].strip(".zip")
    path_to_images = "/host/data/%s" % (images_name)

    # Training Parameters
    n_train_iterations = 4000

    # verbose string
    pr_str = "\n-----------------------------------------------"
    # get images
    print(pr_str)
    print("Downloading Data... ")
    print(pr_str)
    path_download = get_url(url_images, path_to_file)

    # unzip images
    call('mkdir %s && unzip %s -d %s' % \
         (path_to_images, path_download[0], path_to_images), shell=True)

    # class directories
    class_dirs = os.listdir(path_to_images)

    # info message
    print(pr_str)
    print("Retraining Model... ")
    print(pr_str)

    # retrain last layer
    retrain_cmd = 'python3 tensorflow/examples/image_retraining/retrain.py \
    --bottleneck_dir=%sbottlenecks \
    --how_many_training_steps %s \
    --model_dir=%sinception \
    --output_graph=%sretrained_graph.pb \
    --output_labels=%sretrained_labels.txt \
    --image_dir %s' % (path_to_data, n_train_iterations, path_to_data,
                       path_to_data, path_to_data, path_to_images)
    Popen(retrain_cmd, cwd=path_to_tf, shell=True).wait()

    print(pr_str)
    print("Optimizing Model... ")
    print(pr_str)

    # optimize model 1
    opt1_cmd = 'python3 tensorflow/python/tools/optimize_for_inference.py \
    --input=%sretrained_graph.pb \
    --output=%soptimized_retrained_graph.pb \
    --input_names=Mul \
    --output_names=final_result' % (path_to_data, path_to_data)
    Popen(opt1_cmd, cwd=path_to_tf, shell=True).wait()

    # optimize model 2
    opt2_cmd = 'python3 tensorflow/tools/quantization/quantize_graph.py \
    --input=%soptimized_retrained_graph.pb \
    --output=%srounded_retrained_graph.pb \
    --output_node_names=final_result \
    --mode=weights_rounded' % (path_to_data, path_to_data)
    Popen(opt2_cmd, cwd=path_to_tf, shell=True).wait()

    print(pr_str)
    print("Zip and send model ... ")
    print(pr_str)

    # zip final model
    call('cp rounded_retrained_graph.pb graph.pb', shell=True,
         cwd=path_to_data)
    call('cp retrained_labels.txt labels.txt', shell=True, cwd=path_to_data)
    call('zip final_model_%s.zip graph.pb labels.txt' % (images_name),
         shell=True, cwd=path_to_data)

    # send re-trained and optmized model to specified url
    upload_zip(res_file='%sfinal_model_%s.zip' % (path_to_data, images_name),
               ds_id=dataset_id,
               name=images_name,
               filename='final_model_%s.zip' % (images_name),
               desc=desc)




