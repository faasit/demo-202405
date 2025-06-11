from faasit_runtime import function, workflow, Workflow
from faasit_runtime import durable
from faasit_runtime.durable.runtime import DurableRuntime
from faasit_runtime.txn import sagaTask, WithSaga,frontend_recover
import json

USER_ID = 413
@durable
def init(df: DurableRuntime):
    # 打开并读取 hotels.json 文件
    store = df.storage
    hotel = store.get('hotels.json')
    hotels_data = json.loads(hotel)
    # 提取 id 和 rooms 的值
    id_and_rooms = [(hotel['id'], hotel['rooms']) for hotel in hotels_data]

    # 输出结果
    for hotel_id, rooms in id_and_rooms:
        print(f"ID: {hotel_id}, Rooms: {rooms}")
        df.setState(str(hotel_id), rooms)
    df.setState('413', json.dumps([5]))

@durable
def reservation(df: DurableRuntime):
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


@workflow
def safe_resersation(wf:Workflow):
    wf.call('reversehotel',{
        'lambdaId': 'reversehotel',
        'instanceId': 3,
        'selected_totel': [1,2,3]
    })

init = init.export()
reservation = reservation.export()
safe_resersation = safe_resersation.export()