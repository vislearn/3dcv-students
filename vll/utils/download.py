import os
import tarfile
import requests


def download_mnist():
    url = 'https://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz'
    base_path = './data'
    file_name = 'mnist.tar.gz'
    
    # Check if file has been extracted
    if os.path.isdir(os.path.join(base_path, 'mnist')):
        print('MNIST is ready!')
        return
    
    # Check if file exists
    file_path = os.path.join(base_path, file_name)
    if os.path.isfile(file_path):
        print(f'{file_name} already downloaded!')
    else:
        print(f'Downloading {file_name}')
        response = requests.get(url, stream=True)
        with open(file_path, 'wb') as handle:
            for data in response.iter_content():
                handle.write(data)
    
    # Extract file
    print(f'Extract {file_name}')
    tar = tarfile.open(file_path, "r:gz")
    tar.extractall(base_path)
    tar.close()
    os.rename(os.path.join(base_path, 'mnist_png'), os.path.join(base_path, 'mnist'))
    
    print('Done!')
