from faasit_runtime.txn import sagaTask, WithSaga,frontend_recover
from faasit_runtime import FaasitRuntime,create_handler,durable
from faasit_runtime.durable import DurableRuntime
import json

USER_ID = 413

@durable
def safe_reservation(df: DurableRuntime):
    _input = df.input()
    selected = _input.get('selected_totel')

    store = df.storage
    hotel = store.get('hotels.json')
    hotels_data = json.loads(hotel)

    # 提取 id 和 rooms 的值
    id_and_rooms = [(hotel['id'], hotel['rooms']) for hotel in hotels_data]
    safe_reserv = {}
    for hotel_id, rooms in id_and_rooms:
        V_resp = df.getState(str(hotel_id))
        print(f"V_resp: {V_resp}")
        safe_reserv[hotel_id] = V_resp
        for h in hotels_data:
            if str(h['id']) == str(hotel_id):
                h['rooms'] = V_resp
    
    reversed_hotels = df.getState(str(USER_ID))
    if reversed_hotels is None:
        p_reserv = []
    else:
        p_reserv = json.loads(reversed_hotels)

    def reverseHotel(txnID, payload):
        print("reverseHotel")
        for hid, rn in safe_reserv.items():
            if str(hid) in selected:
                p_reserv.append(hid)
                new_rooms = rn - 1
                df.setState(str(hid), new_rooms)
                df.setState(str(USER_ID), json.dumps(p_reserv))
                for h in hotels_data:
                    if str(h['id']) == str(hid):
                        h['rooms'] = new_rooms
        import random
        flag = random.choice([True,False])
        if flag == True:
            return payload
        else:
            print("random error")
            raise Exception("random error")
    def compensateFn(txnID, result):
        pass
    task = sagaTask(reverseHotel, compensateFn)
    result = WithSaga([task], frontend_recover(5))
    result = result(hotels_data)
    return df.output({
        'result': result.result
    })

handler = create_handler(safe_reservation)