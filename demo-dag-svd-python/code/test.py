import os
import time
os.environ["FAASIT_PROVIDER"]="local-once"
from index import handler
inputData = {}
start_time = time.time()
output = handler(inputData)
end_time = time.time()
print(output)
print(f"Execution time: {end_time - start_time} seconds")