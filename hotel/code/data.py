
# string string string string string string float64 float64
class Address:
    def __init__(self, StreetNumber, StreetName, City, State, Country,
                 PostalCode, Lat, Lon):
        self.StreetNumber = StreetNumber
        self.StreetName = StreetName
        self.City = City
        self.State = State
        self.Country = Country
        self.PostalCode = PostalCode
        self.Lat = Lat
        self.Lon = Lon


# string string string string Address
class Hotel:
    def __init__(self, Id, Name, PhoneNumber, Description, Address):
        self.Id = Id
        self.Name = Name
        self.PhoneNumber = PhoneNumber
        self.Description = Description
        self.Address = Address


# float64 float64 float64 string string
class RoomType:
    def __init__(self, BookableRate, TotalRate, TotalRateInclusive, Code,
                 RoomDescription):
        self.BookableRate = BookableRate
        self.TotalRate = TotalRate
        self.TotalRateInclusive = TotalRateInclusive
        self.Code = Code
        self.RoomDescription = RoomDescription


# string string string string RoomType
class RatePlan:
    def __init__(self, HotelId, Code, Indate, Outdate, RoomType):
        self.HotelId = HotelId
        self.Code = Code
        self.Indate = Indate
        self.Outdate = Outdate
        self.RoomType = RoomType


# string float64 float64 float64 float64
class Recommend:
    def __init__(self, HId, HLat, HLon, HRat, HPrice):
        self.HId = HId
        self.HLat = HLat
        self.HLon = HLon
        self.HRat = HRat
        self.HPrice = HPrice


# string string string string int
class Reservation:
    def __init__(self, HotelId, CustomerNam, InDate, OutDate, Number):
        self.HotelId = HotelId
        self.CustomerNam = CustomerNam
        self.InDate = InDate
        self.OutDate = OutDate
        self.Number = Number


# string int32
class Number:
    def __init__(self, HotelId, Num):
        self.HotelId = HotelId
        self.Num = Num


# string string
class User:
    def __init__(self, Username, Password):
        self.Username = Username
        self.Password = Password


# string float64 float64
class Point:
    def __init__(self, Pid, Plat, Plon, Rate):
        self.Pid = Pid
        self.Plat = Plat
        self.Plon = Plon
        self.Rate = Rate

    def Lat(self):
        return self.Plat

    def Lon(self):
        return self.Plon

    def Id(self):
        return self.Pid
    
    def Rate(self):
        return self.Rate


# string interface{}
class RPCInput:
    def __init__(self, Function, Input):
        self.Function = Function
        self.Input = Input


def Tgeo():
    return "bgeo"


def Tflight():
    return "bflight"


def Tfrontend():
    return "bfrontend"


def Tgateway():
    return "bgateway"


def Thotel():
    return "bhotel"


def Torder():
    return "border"


def Tprofile():
    return "bprofile"


def Trate():
    return "brate"


def Trecommendation():
    return "brecommendation"


def Tsearch():
    return "bsearch"


def Tuser():
    return "buser"