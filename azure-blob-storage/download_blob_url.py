import argparse
import os

from azure.storage.blob import BlobClient


def download_anonymously(url, dest_dir='.'):
    blob_client = BlobClient.from_blob_url(url)
    file_name = url.split('/')[-1]
    file_path = os.path.join(dest_dir, file_name)

    # Download the blob to a local file
    with open(file_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())


def main(args):
    download_anonymously(args.url, args.dest_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, help='URL for the Azure blob')
    parser.add_argument('--dest_dir',
                        type=str,
                        default='.',
                        help='local directory where the blob should be stored')
    args = parser.parse_args()
    main(args)