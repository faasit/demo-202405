import os
os.environ['FAASIT_FUNC_NAME']="__executor"
os.environ["FAASIT_PROVIDER"]="local-once"
import json
from index import handler;
import asyncio
inputData = {
    "split": {
        "bundle_size": 1,
        "skew": 1,
        "input": {
            "indent": "Intent.json"
        },
        "output": {
            "bos": 'bos.txt',
            "list_of_indents": 'list_of_indents'
        }
    },
    "train": {
        "input": {
            "indent": "Intent.json",
            "bos": "bos.txt",
            "list_of_indents": "list_of_indents"
        },
        "output": {
            "params": "params.txt"
        }
    }
}
async def main():
    output = await handler(inputData);
    print(json.dumps(output))
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()