import os
import re
import boto3
import requests
import rasterio
import pandas as pd
from rasterio.mask import mask
from dotenv import load_dotenv

from scripts.utils import TMP_PATH

load_dotenv()

# get list of all the available images of certain area
def get_catalogue(start_date, end_date, aoi, data_collection='SENTINEL-2', cloud_cover=10.0):
    uri = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq '{data_collection}' and OData.CSC.Intersects(area=geography'SRID=4326;{aoi}') and ContentDate/Start gt {start_date}T00:00:00.000Z and ContentDate/Start lt {end_date}T00:00:00.000Z and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value lt {cloud_cover})"

    resp = requests.get(uri)

    return resp.json()

# get the s3 container url for images
def get_s3_uri(jsn):
    df = pd.DataFrame.from_dict(jsn['value'])
    
    uri = df[df['Name'].str.contains('MSIL2A')].sort_values(by='PublicationDate', ascending=False)['S3Path'].iloc[0]

    prefix = '/'.join(uri.split('/')[2:])

    return uri, prefix

# download images from s3
def get_s3_content(s3_uri, prefix, bucket='eodata'):
    client = boto3.client(
        's3',
        aws_access_key_id=os.environ['access'],
        aws_secret_access_key=os.environ['sec'],
        region_name='default',
        endpoint_url='https://eodata.dataspace.copernicus.eu'
    )

    pattern = 'GRANULE/[^/]+/IMG_DATA/R10m/(.*?)_TCI_10m.jp2$'

    rslt = client.list_objects(Bucket=bucket, Prefix=prefix)
    
    files = [x['Key'] for x in rslt['Contents'] if re.search(pattern, x['Key'])]

    for x in files:
        tmp = x.split('/')[-1].split('_')[-2]

        print(f'Downloading {tmp}')
        client.download_file(Bucket=bucket, Key=x, Filename=f'{TMP_PATH}/{tmp}.jp2')
    
    return f'{TMP_PATH}/{tmp}.jp2'

# crop out only the required area
def get_roi(img, roi):
    img = rasterio.open(img)
    
    # convert to suitable coordinate format
    msk = rasterio.warp.transform_geom('EPSG:4326', 'EPSG:32645', roi)

    n_img, n_trans = mask(img, [msk], crop=True, filled=True)

    with rasterio.open(f'{TMP_PATH}/roi.png', mode='w', driver='PNG', width=n_img.shape[2], height=n_img.shape[1], count=3, dtype=n_img.dtype, nodata=0) as f:
        f.write(n_img)
    
    return f"{TMP_PATH}/roi.png"
