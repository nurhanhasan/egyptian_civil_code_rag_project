#!/usr/bin/env bash -ex

HOST_PORT="http://localhost:8000"

curl -i -X 'POST' \
  "${HOST_PORT}/api/v1/ask" \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "متى تسقط الدعوى المدنية بالتقادم؟"
}'

curl -i -X 'POST' \
  "${HOST_PORT}/api/v1/ask" \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "ما عقوبة جريمة القتل"             
}'

curl -i -X 'POST' \
  "${HOST_PORT}/api/v1/ask" \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "ما حكم الشرط الجزائي في العقد؟"
}'

curl -i -X 'POST' \
  "${HOST_PORT}/api/v1/ask" \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "ما هي آثار عقد البيع في القانون المدني المصري؟"          
}'

curl -X 'POST' \
  "${HOST_PORT}/api/v1/ask" \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "كيف يتم إثبات الالتزام؟"          
}'

curl -X 'POST' \
  "${HOST_PORT}/api/v1/ask" \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "ما هي شروط صحة الإيجار؟"           
}'

curl -X 'POST' \
  "${HOST_PORT}/api/v1/ask" \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "ما هي أسباب انقضاء الالتزام؟"          
}'

curl -X 'POST' \
  "${HOST_PORT}/api/v1/ask" \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "ما الفرق بين الفسخ والانفساخ؟"          
}'

curl -X 'POST' \
  "${HOST_PORT}/api/v1/ask" \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "ما هي مسؤولية المتبوع عن أعمال التابع؟"          
}'

curl -X 'POST' \
  "${HOST_PORT}/api/v1/ask" \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "متى يكون العقد باطلاً بطلاناً مطلقاً؟"          
}'

curl -X 'POST' \
  "${HOST_PORT}/api/v1/ask" \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "ما هي آثار الإثراء بلا سبب؟"          
}'

curl -X 'POST' \
  "${HOST_PORT}/api/v1/ask" \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "ما هي أحكام التعويض عن الضرر المعنوي؟"          
}'

curl -X 'POST' \
  "${HOST_PORT}/api/v1/ask" \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "كيف تنقضي الملكية؟"          
}'

curl -X 'POST' \
  "${HOST_PORT}/api/v1/ask" \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "ما هي ضمانات عيوب المبيع؟"          
}'

curl -X 'POST' \
  "${HOST_PORT}/api/v1/ask" \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "ما حكم التصرف في ملك الغير؟"          
}'

curl -X 'POST' \
  "${HOST_PORT}/api/v1/ask" \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d '{
  "question":           
}'