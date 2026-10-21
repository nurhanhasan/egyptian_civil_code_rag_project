#!/usr/bin/env bash
set -euo pipefail

HOST_PORT="${HOST_PORT:-http://localhost:8000}"
COUNT="${1:-3}"

questions=(
  "متى تسقط الدعوى المدنية بالتقادم؟"
  "ما عقوبة جريمة القتل"
  "ما حكم الشرط الجزائي في العقد؟"
  "ما هي آثار عقد البيع في القانون المدني المصري؟"
  "كيف يتم إثبات الالتزام؟"
  "ما هي شروط صحة الإيجار؟"
  "ما هي أسباب انقضاء الالتزام؟"
  "ما الفرق بين الفسخ والانفساخ؟"
  "ما هي مسؤولية المتبوع عن أعمال التابع؟"
  "متى يكون العقد باطلاً بطلاناً مطلقاً؟"
  "ما هي آثار الإثراء بلا سبب؟"
  "ما هي أحكام التعويض عن الضرر المعنوي؟"
  "كيف تنقضي الملكية؟"
  "ما هي ضمانات عيوب المبيع؟"
  "ما حكم التصرف في ملك الغير؟"
)

if command -v shuf >/dev/null 2>&1; then
  selected="$(printf '%s\n' "${questions[@]}" | shuf -n "${COUNT}")"
else
  selected="$(
    printf '%s\n' "${questions[@]}" |
      awk 'BEGIN { srand() } { print rand() "\t" NR "\t" $0 }' |
      sort -n |
      head -n "${COUNT}" |
      cut -f3-
  )"
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

pids=()
selected_questions=()
idx=0

while IFS= read -r question; do
  [ -n "${question}" ] || continue
  idx=$((idx + 1))
  selected_questions[$idx]="${question}"
  curl -i -X POST \
    "${HOST_PORT}/api/v1/ask" \
    -H 'accept: */*' \
    -H 'Content-Type: application/json' \
    --data-binary "$(printf '{"question":"%s"}' "${question}")" \
    >"${tmpdir}/${idx}.out" 2>"${tmpdir}/${idx}.err" &
  pids[$idx]=$!
done <<< "${selected}"

status=0
for ((i = 1; i <= idx; i++)); do
  wait "${pids[$i]}" || status=1
done

for ((i = 1; i <= idx; i++)); do
  echo "Question: ${selected_questions[$i]}"
  cat "${tmpdir}/${i}.out"
  if [ -s "${tmpdir}/${i}.err" ]; then
    cat "${tmpdir}/${i}.err"
  fi
  echo
done

exit "${status}"
