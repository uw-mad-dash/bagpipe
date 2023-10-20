import time
import boto3


class uploadFile(object):
    """
    Initialize with the bucket name
    If the bucket doesn't exist than create the bucket
    """

    def __init__(self, bucket_name):
        """
        create s3 resource and creates the bucket if the given bucket name
        doesn't exist
        """
        self.bucket_name = bucket_name
        self.resource = boto3.client("s3")
        bucket_response = self.resource.list_buckets()
        all_buckets = [bucket["Name"] for bucket in bucket_response["Buckets"]]
        if self.bucket_name not in all_buckets:
            # Creat the bucket if doesn't exist
            resp = self.resource.create_bucket(Bucket=self.bucket_name)
            # print(resp)
        return None

    def push_file(self, input_path, out_file_path):
        print("Start Uploading")
        self.resource.upload_file(input_path, self.bucket_name, out_file_path)
        print("Upload Successfully")

    def pull_file(self, input_path, out_file_path):
        self.resource.download_file(self.bucket_name, input_path, out_file_path)


if __name__ == "__main__":
    # usage
    # by default buckets will be created in north virginia
    file_uploader = uploadFile("large-scale-compression")
    with open("test.txt", "a") as out_f:
        out_f.write("Test time = {}".format(time.time()))
    file_uploader.push_file("./test.txt", "/test/test.txt")
