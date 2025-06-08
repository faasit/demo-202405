import json
import time
from faasit_runtime import function, FaasitRuntime

def newHotelIndex(frt: FaasitRuntime) -> dict:
	file_path = 'tmp/Hotel_info.json'
	hotel = frt.storage.get("Hotel_info.json")
	hotel = json.loads(hotel)
	
	# with open(file_path, 'r') as file:
		# hotel = json.load(file)
	
	hotel_id_info = {str(hotel_info['HotelId']): 0 for hotel_info in hotel}
	
	return hotel_id_info


@function
def BaseReserveHotel(frt: FaasitRuntime):
	start_time = time.time()
	params = frt.input()
	ia_hotel_id = params['input']['hotel_id']
	ia_reserve_date = params['input']['reserve_date']
   
	max_capacity = 3
	compute_time, download_time, communication_cnt = 0, 0, 0

	start_download_time = time.time()
	store = frt.storage
	hotel_id_reserve : dict = store.get(ia_hotel_id, src_stage='rate')
	# hotel_id_reserve : dict = md.get_object('stage2', ia_hotel_id)
	reverse_data = store.get(ia_reserve_date, src_stage='search')
	# reserve_data = md.get_object('stage0', ia_reserve_date)
	end_download_time = time.time()
	download_time = end_download_time - start_download_time

	start_compute_time = time.time()
	hotel_id_info = newHotelIndex(frt)
	keys = hotel_id_reserve.keys()		
				  
	for key in keys:
		hotel_id = hotel_id_reserve[key]
		hotel_cap = hotel_id_info[str(hotel_id)]
		if hotel_cap <= max_capacity:
			hotel_id_info[str(hotel_id)] += 1
		else:
			pass
		communication_cnt += 1
	end_compute_time = time.time()
	compute_time = end_compute_time - start_compute_time
	
	end_time = time.time()
	process_time = end_time - start_time
	
	return_val = {
		'communication': communication_cnt,
		'process_time': process_time,
		'download_time': download_time,
		'compute_time': compute_time,
		'communication_time': 0,
	}
	
	return {
		'statusCode': 200,
		'body': json.dumps(return_val)
	}

baseReserveHotel = BaseReserveHotel.export()