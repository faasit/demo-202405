import boto3

# DynamoDB 本地服务地址
dynamodb = boto3.client('dynamodb', endpoint_url='http://192.168.82.40:8000')

def create_log_table(lambda_id):
    table_name = f"{lambda_id}-log"
    try:
        dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {"AttributeName": "InstanceId", "KeyType": "HASH"},  # 分区键
                {"AttributeName": "StepNumber", "KeyType": "RANGE"}  # 排序键
            ],
            AttributeDefinitions=[
                {"AttributeName": "InstanceId", "AttributeType": "S"},
                {"AttributeName": "StepNumber", "AttributeType": "N"}
            ],
            ProvisionedThroughput={
                "ReadCapacityUnits": 5,
                "WriteCapacityUnits": 5
            }
        )
        print(f"Table {table_name} created successfully.")
    except dynamodb.exceptions.ResourceInUseException:
        print(f"Table {table_name} already exists.")

def create_intent_table(lambda_id):
    table_name = f"{lambda_id}-collector"
    try:
        dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {"AttributeName": "InstanceId", "KeyType": "HASH"}
            ],
            AttributeDefinitions=[
                {"AttributeName": "InstanceId", "AttributeType": "S"}
            ],
            ProvisionedThroughput={
                "ReadCapacityUnits": 5,
                "WriteCapacityUnits": 5
            }
        )
        print(f"Table {table_name} created successfully.")
    except dynamodb.exceptions.ResourceInUseException:
        print(f"Table {table_name} already exists.")

def create_local_table(lambda_id):
    table_name = f"{lambda_id}-local"
    try:
        dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {"AttributeName": "K", "KeyType": "HASH"}
            ],
            AttributeDefinitions=[
                {"AttributeName": "K", "AttributeType": "S"}
            ],
            ProvisionedThroughput={
                "ReadCapacityUnits": 5,
                "WriteCapacityUnits": 5
            }
        )
        print(f"Table {table_name} created successfully.")
    except dynamodb.exceptions.ResourceInUseException:
        print(f"Table {table_name} already exists.")

def delete_all_tables():
    """
    删除 DynamoDB 中的所有表
    """
    try:
        # 获取所有表名
        response = dynamodb.list_tables()
        tables = response.get('TableNames', [])
        if not tables:
            print("No tables found to delete.")
            return

        for table_name in tables:
            print(f"Deleting table: {table_name}...")
            dynamodb.delete_table(TableName=table_name)

            # 等待表删除完成
            waiter = dynamodb.get_waiter('table_not_exists')
            waiter.wait(TableName=table_name)
            print(f"Table {table_name} deleted successfully.")

    except Exception as e:
        print(f"Error while deleting tables: {e}")

def main():
    lambda_id = "hotelReserve"
    
    # 删除所有表
    delete_all_tables()
    
    # 创建需要的表
    create_log_table(lambda_id)
    create_intent_table(lambda_id)
    create_local_table(lambda_id)

if __name__ == "__main__":
    main()