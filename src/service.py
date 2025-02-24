from abc import ABC, abstractmethod
import random
from typing import Optional, List
import redis
import pandas as pd


class DataStorage(ABC):
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_data(self, df: pd.DataFrame) -> None:
        pass


class WeightStorage(ABC):
    @abstractmethod
    def get_weight(self, image_url: str) -> Optional[float]:
        pass

    @abstractmethod
    def set_weight(self, image_url: str, weight: float) -> None:
        pass

    @abstractmethod
    def get_show_count(self, image_url: str) -> int:
        pass

    @abstractmethod
    def increment_show_count(self, image_url: str) -> None:
        pass


class CSVDataStorage(DataStorage):
    def __init__(self, csv_path: str = 'data.csv'):
        self.csv_path = csv_path

    def load_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.csv_path, sep=';', dtype=str, on_bad_lines='warn')
            df['needed_amount_of_shows'] = pd.to_numeric(df['needed_amount_of_shows'], errors='coerce')
            category_cols = [col for col in df.columns if col.startswith('category')]
            df[category_cols] = df[category_cols].fillna('')
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV: {e}")

    def save_data(self, df: pd.DataFrame) -> None:
        df.to_csv(self.csv_path, sep=';', index=False)


class RedisWeightStorage(WeightStorage):
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)

    def get_weight(self, image_url: str) -> Optional[float]:
        weight = self.client.get(f"weight:{image_url}")
        return float(weight) if weight is not None else None

    def set_weight(self, image_url: str, weight: float) -> None:
        self.client.set(f"weight:{image_url}", str(weight))

    def get_show_count(self, image_url: str) -> int:
        return int(self.client.get(f"shows:{image_url}") or 0)

    def increment_show_count(self, image_url: str) -> None:
        self.client.incr(f"shows:{image_url}")


class ImageService:
    def __init__(self, data_storage: DataStorage, weight_storage: WeightStorage):
        self.data_storage = data_storage
        self.weight_storage = weight_storage
        self.df = self.data_storage.load_data()

    def _calculate_weight(self, image_row: pd.Series) -> float:
        image_url = image_row['Image_URL']
        show_count = self.weight_storage.get_show_count(image_url)
        weight = 1.0 / (show_count / 100 + 1)

        needed_shows = image_row['needed_amount_of_shows']
        if pd.isna(needed_shows):
            needed_shows = 0

        weight *= (needed_shows / 100 + 1)
        return weight

    def get_image(self, categories: Optional[List[str]] = None) -> Optional[str]:
        """Получает URL изображения на основе весов и категорий.
            Функция фильтрует изображения из DataFrame по количеству оставшихся показов и,
            если указаны категории, по совпадению с ними. Затем вычисляет или извлекает веса
            для каждого изображения, выбирает одно с учётом весов, обновляет данные и возвращает его URL.
            Args:
                categories (Optional[List[str]], optional): Список категорий для фильтрации изображений.
                    Если None, фильтрация по категориям не применяется. Defaults to None.
            Returns:
                Optional[str]: URL выбранного изображения или None, если подходящих изображений нет.
            Notes:
                Вес изображения зависит от количества показов (`shows`) и необходимых показов
                  (`needed_amount_of_shows`).
                После выбора изображения обновляются его вес и количество оставшихся показов,
                  данные сохраняются в хранилище.
            """
        filtered_df = self.df[self.df['needed_amount_of_shows'] > 0].copy()

        if categories:
            category_columns = [col for col in self.df.columns if col.startswith('category')]
            mask_categories = filtered_df[category_columns].isin(categories).any(axis=1)
            filtered_df = filtered_df[mask_categories]

        if filtered_df.empty:
            return None

        weighted_images = []
        for _, row in filtered_df.iterrows():
            image_url = row['Image_URL']
            weight = self.weight_storage.get_weight(image_url)
            if weight is None:
                weight = self._calculate_weight(row)
                self.weight_storage.set_weight(image_url, weight)
            weighted_images.append((row, weight))

        if not weighted_images:
            return None

        images, weights = zip(*weighted_images)
        selected_row = random.choices(images, weights=weights, k=1)[0]
        selected_url = selected_row['Image_URL']

        self.weight_storage.increment_show_count(selected_url)

        idx = self.df[self.df['Image_URL'] == selected_url].index[0]
        self.df.at[idx, 'needed_amount_of_shows'] -= 1

        new_weight = self._calculate_weight(self.df.loc[idx])
        self.weight_storage.set_weight(selected_url, new_weight)

        self.data_storage.save_data(self.df)
        return selected_url
