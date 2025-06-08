from faasit_runtime import workflow, Workflow
import random

request_id = "000000"
use_redis_when_remote = True

@workflow
def travelReservation(wf: Workflow):
    s0 = wf.call("search", {
        'output': {
            'reserve_loc': f'{request_id}-Hotel_Reservation/stage0/reserve_loc', 
            'reserve_date': f'{request_id}-Hotel_Reservation/stage0/reserve_date'
        },
        "request_cnt": 100
    })
    s1 = wf.call("geo", {"s0": s0, 'input': {
            'hotel_info' : f'{request_id}-Hotel_Reservation/stage0/hotel_info',
            'reserve_loc': f'{request_id}-Hotel_Reservation/stage0/reserve_loc',
        },
        'output': {
            'hotel_id': f'{request_id}-Hotel_Reservation/stage1/hotel_id'
        },
        'kNearest': random.randint(1, 10)})
    s2 = wf.call("rate", {
        "s1": s1,
        'input': {
            'hotel_info' : f'{request_id}-Hotel_Reservation/stage0/hotel_info', 
            'hotel_id': f'{request_id}-Hotel_Reservation/stage1/hotel_id'
        },
        'output': {
            'hotel_id': f'{request_id}-Hotel_Reservation/stage2/hotel_id' 
        },
    })
    s3 = wf.call("hotel", {
        "s0": s0,
        "s2": s2, 
        'input':{
            'hotel_id': f'{request_id}-Hotel_Reservation/stage2/hotel_id',
            'reserve_date': f'{request_id}-Hotel_Reservation/stage0/reserve_date'
        },
    })
    return s3
    
travelReservation = travelReservation.export()