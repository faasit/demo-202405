import json
import time
from faasit_runtime import FaasitRuntime, function

def newHotelRate(frt: FaasitRuntime):
	file_path = 'tmp/Hotel_info.json'
	hotel = frt.storage.get("Hotel_info.json")
	hotel = json.loads(hotel)
	# with open(file_path, 'r') as file:
		# hotel = json.load(file)

	info = {hotel_info['HotelId']: hotel_info['Rate'] for hotel_info in hotel}
	return info

@function
def GetRates(frt: FaasitRuntime):
	start_time = time.time()
	params = frt.input()
	ia = params['input']['hotel_id']
	oa = params['output']['hotel_id']
	communication_cnt = 0
	compute_time, communication_time, download_time = 0, 0, 0

	start_download_time = time.time()
	store = frt.storage
	hotel_id_info : dict = store.get(ia, src_stage='geo')
	# hotel_id_info : dict = md.get_object('stage1', ia)
	end_download_time = time.time()
	download_time += end_download_time - start_download_time
	
	start_compute_time = time.time()
	hotel_rate = newHotelRate(frt)
	hotel_book_info = {}
	keys = hotel_id_info.keys()

	for key in keys:
		hotel_id = hotel_id_info[key]
		hotel_book = max(hotel_id, key=lambda i:hotel_rate[i])
		
		reserve_cnt = key.split('-')[-1]
		hotel_book_cnt = f'{oa}-{reserve_cnt}'
		hotel_book_info[hotel_book_cnt] = hotel_book
		communication_cnt += 1

	end_compute_time = time.time()
	compute_time += end_compute_time - start_compute_time

	end_time = time.time()
	process_time = end_time - start_time

	start_commnication_time = time.time()
	store.put(oa, hotel_book_info, dest_stages=['hotel'])
	# md.output(['stage3'], oa, hotel_book_info)
	end_communication_time = time.time()
	communication_time = end_communication_time - start_commnication_time
			
	return_val = {
        'communication': communication_cnt,
		'process_time': process_time,
		'download_time': download_time,
		'compute_time': compute_time,
		'upload_time': communication_time,
		#'debug_msg': debug_msg
    }
	
	return {
        'statusCode': 200,
        'body': json.dumps(return_val)
    }

getRates = GetRates.export()