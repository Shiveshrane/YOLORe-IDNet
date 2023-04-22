import exiftool
import requests
from PIL import Image
import json
from typing import Dict, List, Tuple
import folium
import io
from PIL import Image
from folium.plugins import BeautifyIcon

class VideoMetadataExtractor:
    def __init__(self, file_path: str) -> None:
        with open(file_path, 'r') as f:
            self.video_urls = f.read().splitlines()
        self.metadata = self._extract_metadata_from_videos()

    def _extract_metadata_from_videos(self) -> Dict[str, Dict[str, float]]:
        metadata_dict = {}
        with exiftool.ExifTool() as et:
            for i, video_url in enumerate(self.video_urls):
                metadata = et.execute('-G', '-j', video_url)
                if metadata:
                    metadata = json.loads(metadata)[0]
                    metadata_block = metadata
                    latitude = metadata_block['Composite:GPSLatitude']
                    longitude = metadata_block['Composite:GPSLongitude']
                    if latitude is not None and longitude is not None:
                        metadata_dict[f"{i}"] = {
                            "latitude": latitude,
                            "longitude": longitude
                        }
        return metadata_dict


    def _convert_to_decimal_degrees(self, coordinates: Tuple[float, float, float]) -> float:
        degrees = coordinates[0]
        minutes = coordinates[1]
        seconds = coordinates[2]
        decimal_degrees = degrees + (minutes / 60) + (seconds / 3600)
        return decimal_degrees

    def get_metadata_by_camera_id(self, camera_id: str) -> Dict[str, float]:
        for video_num, metadata in self.metadata.items():
            if camera_id in video_num:
                return metadata
        raise ValueError(f"No metadata found for camera {camera_id}")


    def save_trajectory_map(self, locations: List[Dict[str, float]], filename: str) -> None:
        map_center = [sum([loc['latitude'] for loc in locations])/len(locations),
                      sum([loc['longitude'] for loc in locations])/len(locations)]
        map_obj = folium.Map(location=map_center, zoom_start=100)

        folium.PolyLine(locations=[(loc['latitude'], loc['longitude']) for loc in locations],
                        color='blue',
                        weight=3).add_to(map_obj)

        for i, loc in enumerate(locations):
            folium.Marker(location=(loc['latitude'], loc['longitude']),
                          icon=folium.Icon(color='red', icon='person-circle-exclamation', prefix='fa'),
                          popup=f'Location {i+1}').add_to(map_obj)
        map_obj.save(filename+'.html')
