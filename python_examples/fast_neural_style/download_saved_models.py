import urllib.request
import zipfile


if __name__ == "__main__":
    url = "https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=1"

    u = urllib.request.urlopen(url)
    data = u.read()
    u.close()

    with open('saved_models.zip', "wb") as f:
        f.write(data)

    zip_ref = zipfile.ZipFile('saved_models.zip', 'r')
    zip_ref.extractall('.')
    zip_ref.close()
