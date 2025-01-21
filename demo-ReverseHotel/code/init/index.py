from faasit_runtime import function,create_handler
from faasit_runtime import durable
from faasit_runtime.durable import DurableRuntime

import json
from table_manage import main
@durable
def init(df: DurableRuntime):
    # 打开并读取 hotels.json 文件
    main()
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

handler = create_handler(init)