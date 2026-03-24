$body = @{
    schedule_time   = "string"
    topics          = @("Ottawa","Senators","Hockey")
    max_duration_min = 7
    voice           = "en_US"
} | ConvertTo-Json

$response = Invoke-RestMethod -Method Post `
    -Uri "http://localhost:8000/users" `
    -Headers @{
        "accept" = "application/json"
        "Content-Type" = "application/json"
    } `
    -Body $body

$response





