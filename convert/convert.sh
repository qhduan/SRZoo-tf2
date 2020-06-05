set -e
mkdir -p outputs
python convert.py "./models/edsr_baseline_x2.pb" "outputs/edsr_baseline_x2"

python convert.py "./models/edsr_baseline_x3.pb" "outputs/edsr_baseline_x3"

python convert.py "./models/edsr_baseline_x4.pb" "outputs/edsr_baseline_x4"

python convert.py "./models/edsr_x2.pb" "outputs/edsr_x2"

python convert.py "./models/edsr_x3.pb" "outputs/edsr_x3"

python convert.py "./models/edsr_x4.pb" "outputs/edsr_x4"

python convert.py "./models/eusr_x2.pb" "outputs/eusr_x2"

python convert.py "./models/eusr_x4.pb" "outputs/eusr_x4"

python convert.py "./models/eusr_x8.pb" "outputs/eusr_x8"

python convert.py "./models/dbpn_x2.pb" "outputs/dbpn_x2"

python convert.py "./models/dbpn_x4.pb" "outputs/dbpn_x4"

python convert.py "./models/dbpn_x8.pb" "outputs/dbpn_x8"

python convert.py "./models/rcan_x2.pb" "outputs/rcan_x2"

python convert.py "./models/rcan_x3.pb" "outputs/rcan_x3"

python convert.py "./models/rcan_x4.pb" "outputs/rcan_x4"

python convert.py "./models/rcan_x8.pb" "outputs/rcan_x8"

python convert.py "./models/msrn_x2.pb" "outputs/msrn_x2"

python convert.py "./models/msrn_x3.pb" "outputs/msrn_x3"

python convert.py "./models/msrn_x4.pb" "outputs/msrn_x4"

python convert.py "./models/4pp_eusr_pirm_x4.pb" "outputs/4pp_eusr_pirm_x4"

python convert.py "./models/esrgan_x4.pb" "outputs/esrgan_x4"

python convert.py "./models/rrdb_x4.pb" "outputs/rrdb_x4"

python convert.py "./models/carn_x2.pb" "outputs/carn_x2"

python convert.py "./models/carn_x3.pb" "outputs/carn_x3"

python convert.py "./models/carn_x4.pb" "outputs/carn_x4"

python convert.py "./models/frsr_x2.pb" "outputs/frsr_x2"

python convert.py "./models/frsr_x3.pb" "outputs/frsr_x3"

python convert.py "./models/frsr_x4.pb" "outputs/frsr_x4"

python convert.py "./models/natsr_x4.pb" "outputs/natsr_x4"

