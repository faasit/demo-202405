import json
import os
os.environ["FAASIT_PROVIDER"]="local-once"
os.environ['LOCAL_STORAGE_DIR']='../../data'
from index import handler
output = handler({
    'lambdaId': 'reverseHotel',
    'instanceId': 1,
    'selected_totel': ['1','2']
})
print(output)