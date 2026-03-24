$body = @{
    user_id = 1   # 👈 replace with the actual user ID
} | ConvertTo-Json

$response = Invoke-RestMethod -Method Post `
    -Uri "http://localhost:8001/generate_episode" `
    -Headers @{
        "accept" = "application/json"
        "Content-Type" = "application/json"
    } `
    -Body $body

$response
