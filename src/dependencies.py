from typing import Annotated
from fastapi import Depends
from service import ImageService, CSVDataStorage, RedisWeightStorage

img_service = Annotated[ImageService, Depends(
    lambda: ImageService(data_storage=CSVDataStorage(csv_path='data.csv'), weight_storage=RedisWeightStorage()))]
