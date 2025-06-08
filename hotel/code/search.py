import random
import json
import time
from faasit_runtime import FaasitRuntime, function

def get_random_day(month: int, start: int):
    if month in [1, 3, 5, 7, 8, 10, 12]:
        day = random.randint(start, 31)
    elif month in [4, 6, 9, 11]:
        day = random.randint(start, 30)
    else:
        day = random.randint(start, 28)
    return day

def get_random_date():
    in_month = random.randint(1, 12)
    out_month = random.randint(in_month, 12)
    in_day, out_day = 0, 0

    if in_month == out_month:
        in_day = get_random_day(in_month, 1)
        out_day = get_random_day(out_month, in_day)
    else:
        in_day = get_random_day(in_month, 1)
        out_day = get_random_day(out_month, 1)
    return [[in_month, in_day], [out_month, out_day]]

@function
def Search_hotel(frt: FaasitRuntime):
    start_time = time.time()
    params = frt.input()
    random.seed(42)

    oa_reserve_loc = params['output']['reserve_loc']
    oa_reserve_date = params['output']['reserve_date']
    reserve_cnt = 0

    start_random_time = time.time()
    commpute_time = 0
    communication_time = 0
    duration = 2
    max_request_cnt = params['request_cnt']
    reserve_loc = {}
    reserve_date = {}

    while (True): 
        if reserve_cnt > max_request_cnt:
            break

        Lat = 38.0235 + (random.randint(0, 481) - 240.5) / 1000.0
        Lon = -122.095 + (random.randint(0, 325) - 157.0) / 1000.0
        Loc = [Lat, Lon]
        Date = get_random_date()
        
        reserve_loc_cnt = f'{oa_reserve_loc}-{reserve_cnt}'
        reserve_date_cnt = f'{oa_reserve_date}-{reserve_cnt}'
        reserve_loc[reserve_loc_cnt] = Loc
        reserve_date[reserve_date_cnt] = Date
        reserve_cnt += 1

    end_random_time = time.time()
    commpute_time += end_random_time - start_random_time
    
    start_commnication_time = time.time()
    store = frt.storage
    store.put(oa_reserve_loc, reserve_loc, dest_stages=['geo'])
    store.put(oa_reserve_date, reserve_date, dest_stages=['hotel'])
    # md.output(['stage1'], oa_reserve_loc, reserve_loc)
    # md.output(['stage3'], oa_reserve_date, reserve_date)
    end_commnication_time = time.time()
    communication_time += end_commnication_time - start_commnication_time

    end_time = time.time()
    process_time = end_time - start_time
    
    return_val = {
        'communication_cnt': reserve_cnt,
        'process_time': process_time,
        'compute_time': commpute_time,
        'upload_time': communication_time,
        #'update': True,
    }
    
    return {
        'statusCode': 200,
        'body': json.dumps(return_val)
    }

search_hotel = Search_hotel.export()