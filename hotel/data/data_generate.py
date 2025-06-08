import json
import random

def generate_hotel_info():
    hotel_info_path = 'Hotel_info.json'
    hotel_info = []
    max_hotel_num = 10000

    for i in range(0, max_hotel_num):
        HotelId = i
        Lat = 38.0235 + (random.randint(0, 481) - 240.5) / 1000.0
        Lon = -122.095 + (random.randint(0, 325) - 157.0) / 1000.0
        Rate = random.randint(0, 5)

        data = {
            "HotelId" : HotelId,
            "Lat" : Lat,
            "Lon" : Lon,
            "Rate" : Rate,
        }

        hotel_info.append(data)
    
    with open(hotel_info_path, 'w') as file:
        json.dump(hotel_info, file, indent=2)

if __name__ == '__main__':
    generate_hotel_info()