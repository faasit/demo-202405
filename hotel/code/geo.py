import json
import time
from faasit_runtime import function, FaasitRuntime

class Point:
	def __init__(self, Pid, Plat, Plon, Rate):
		self.Pid = Pid
		self.Plat = Plat
		self.Plon = Plon
		self.Rate = Rate

	def Lat(self):
		return self.Plat

	def Lon(self):
		return self.Plon

	def Id(self):
		return self.Pid
	
	def Rate(self):
		return self.Rate


def newHotelIndex(frt: FaasitRuntime) -> list:
	file_path = 'tmp/Hotel_info.json'
	hotel = frt.storage.get("Hotel_info.json")
	hotel = json.loads(hotel)
	# with open(file_path, 'r') as file:
		# hotel = json.load(file)

	info = [Point(hotel_info['HotelId'], hotel_info['Lat'], 
						  hotel_info['Lon'], hotel_info['Rate']) for hotel_info in hotel]
	
	return info


def getNearbyPoints(lat, lon, kNearest: int, points):
	distance = [(p, ((lat - p.Lat())*10)**2 + ((lon - p.Lon())*10)**2) for p in points]
	distance.sort(key=lambda item:item[1])
	return [pair[0].Id() for pair in distance[:kNearest]]


@function
def Nearby(frt: FaasitRuntime):
	start_time = time.time()
	params = frt.input()
	store = frt.storage
	ia = params['input']['reserve_loc']
	oa = params['output']['hotel_id']
	kNearest = params['kNearest']
	communication_cnt = 0
	compute_time, download_time, communication_time = 0, 0, 0

	start_download_time = time.time()
	reserve_loc : dict = store.get(ia, src_stage='search')
	end_download_time = time.time()
	download_time = end_download_time - start_download_time
 
	start_compute_time = time.time()
	points_info = newHotelIndex(frt)
	hotel_info = {}
	
	keys = reserve_loc.keys()

	for key in keys:
		reserve_cnt = key.split('-')[-1]
		Lat, Lon = reserve_loc[key]
		
		points = getNearbyPoints(Lat, Lon, kNearest, points_info)

		hotel_id_cnt = f'{oa}-{reserve_cnt}'
		
		hotel_info[hotel_id_cnt] = points
		
		communication_cnt += 1
	end_compute_time = time.time()
	compute_time += end_compute_time - start_compute_time

	start_commnication_time = time.time()
	store.put(oa, hotel_info, dest_stages=['rate'])
	# md.output(['stage2'], oa, hotel_info)
	end_communication_time = time.time()
	communication_time = end_communication_time - start_commnication_time

	end_time = time.time()
	process_time = end_time - start_time
			
	return_val = {
		'communication': communication_cnt,
		'process_time': process_time,
		'download_time': download_time,
		'compute_time': compute_time,
		'upload_time': communication_time,
	}
	
	return frt.output({
		'statusCode': 200,
		'body': json.dumps(return_val)
	})

nearby = Nearby.export()
