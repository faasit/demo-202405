from faasit_runtime import function, FaasitRuntime
from faasit_runtime.txn import sagaTask, WithSaga, frontend_recover
import random


@function
def handler(frt: FaasitRuntime):

    results = []
    logs = []
    def reverseHotel(txnID, payload):
        logs.append("trying to reverse hotel...")
        if 'reverseHotel' not in results:
            results.append("reverseHotel")
        flag = random.choice([True,False])
        if flag == True:
            logs.append("reversed hotel successfully")
            return payload
        else:
            print("random error")
            raise Exception("random error")
    def compensateHotel(txnID, result):
        logs.append("compensating hotel...")
        results.remove("reverseHotel")
    

    def reverseFlight(txnID, payload):
        logs.append("trying to reverse flight...")
        if 'reverseFlight' not in results:
            results.append("reverseFlight")
        flag = random.choice([True,False])
        if flag == True:
            logs.append("reversed flight successfully")
            return payload
        else:
            print("random error")
            raise Exception("random error")
    def compensateFlight(txnID, result):
        logs.append("compensating flight...")
        results.remove("reverseFlight")
    
    def reverseCar(txnID, payload):
        logs.append("trying to reverse car...")
        if 'reverseCar' not in results:
            results.append("reverseCar")
        flag = random.choice([True,False])
        if flag == True:
            logs.append("reversed car successfully")
            return payload
        else:
            print("random error")
            raise Exception("random error")
    def compensateCar(txnID, result):
        logs.append("compensating car...")
        results.remove("reverseCar")

    task_hotel = sagaTask(reverseHotel, compensateHotel)
    task_flight = sagaTask(reverseFlight, compensateFlight)
    task_car = sagaTask(reverseCar, compensateCar)
    result = WithSaga([task_hotel, task_flight, task_car], frontend_recover(5))
    result = result("payload")
    return frt.output({
        'result': results,
        'logs': logs
    })

handler = handler.export()