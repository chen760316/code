Start the main method, and then send a POST request via HTTP to the interface 127.0.0.1:19123/rds with the following payload:

```
{
  "table": {
    "path": "test_data/flights_10000_change.csv",
    "columnsType": {
      "PID": "int",
      "YEAR": "int",
      "MONTH": "int",
      "DAY": "int",
      "DAY_OF_WEEK": "int",
      "AIRLINE": "string",
      "FLIGHT_NUMBER": "int",
      "ORIGIN_AIRPORT": "string",
      "DESTINATION_AIRPORT": "string",
      "SCHEDULED_DEPARTURE": "int",
      "DEPARTURE_TIME": "float",
      "DEPARTURE_DELAY": "float",
      "SCHEDULED_TIME": "float",
      "TAXI_OUT": "float",
      "WHEELS_OFF": "float",
      "ELAPSED_TIME": "float",
      "DISTANCE": "int",
      "AIR_TIME": "float",
      "TAXI_IN": "float",
      "ARRIVAL_TIME": "float",
      "ARRIVAL_DELAY": "float",
      "SCHEDULED_ARRIVAL": "int",
      "CANCELLATION_REASON": "string",
      "DIVERTED": "int",
      "CANCELLED": "int",
      "AIR_SYSTEM_DELAY": "float",
      "SECURITY_DELAY": "float",
      "WEATHER_DELAY": "int",
      "AIRLINE_DELAY": "float",
      "LATE_AIRCRAFT_DELAY": "float"
    }
  },
  "support": 0.00002,
  "confidence": 0.95
}
```

**Supported data types**: `string`, `int64`, `float64`, `bool`.

