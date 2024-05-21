from enum import Enum

class RequestType(Enum):
    UserLogin = 1
    UpdateReview = 2
    Recommend = 3
    Search = 4